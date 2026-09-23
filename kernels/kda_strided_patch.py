#!/usr/bin/env python3
"""Fork patch 0021: the KDA recurrence reads q, k, v and beta where they already are (no copies inside the CUDA graphs).

In a decode / draft-verify step the KDA layer's q, k, v and beta are column slices of ONE projection output
([tokens, 3*P + H + 2*D]); the short conv runs in place on that buffer. `fused_recurrent_kda` then calls
`.contiguous()` on each of them, which for more than one token is four copy kernels per KDA layer (34 layers, every
step, inside the graphs: about 12 us of GPU time per layer on the 170HX, ~0.4 ms per step over the pipeline).

The Triton kernel only needs the distance between two tokens; inside a token the layout is already dense. So:

  VLLM_KDA_STRIDED=1   (default = upstream)
    - kernel `fused_recurrent_gated_delta_rule_fwd_kernel`: four new constexpr parameters STRIDE_{Q,K,V,BETA}_TOK
      (0 = dense layout, the upstream pointer arithmetic, compiled exactly as before);
    - `fused_recurrent_kda`: a tensor whose layout is [1, tokens, (heads,) dim] with a dense token payload is handed
      over as it is, with its token stride; anything else is made contiguous as before.

Same arithmetic on the same values, only the load addresses differ, so outputs and states are bit-identical
(test_kda_strided_kernel.py checks that on a GPU). Usage: kda_strided_patch.py check|apply|revert"""
import os, shutil, sys, py_compile
ROOT = os.environ.get("FLA_OPS_DIR", "/usr/local/lib/python3.12/dist-packages/vllm/third_party/flash_linear_attention/ops")
F_KERNEL = os.path.join(ROOT, "fused_recurrent.py")
F_KDA = os.path.join(ROOT, "kda.py")
SUFFIX = ".bak0021"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
TAG = "patch 0021"

KERNEL_REPL = [
 ('''    SAFE_GATE: tl.constexpr,  # bounded gate variant (only branch implemented)
    LOWER_BOUND: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
''',
  '''    SAFE_GATE: tl.constexpr,  # bounded gate variant (only branch implemented)
    LOWER_BOUND: tl.constexpr,
    # patch 0021: distance between two tokens of q / k / v / beta in elements; 0 = dense layout (upstream arithmetic)
    STRIDE_Q_TOK: tl.constexpr = 0,
    STRIDE_K_TOK: tl.constexpr = 0,
    STRIDE_V_TOK: tl.constexpr = 0,
    STRIDE_BETA_TOK: tl.constexpr = 0,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
'''),
 ('''    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
''',
  '''    # patch 0021: token strides (dense layout when the constexpr is 0)
    if STRIDE_Q_TOK > 0:
        p_q = q + bos * STRIDE_Q_TOK + i_h * K + o_k
    else:
        p_q = q + (bos * H + i_h) * K + o_k
    if STRIDE_K_TOK > 0:
        p_k = k + bos * STRIDE_K_TOK + i_h * K + o_k
    else:
        p_k = k + (bos * H + i_h) * K + o_k
    if STRIDE_V_TOK > 0:
        p_v = v + bos * STRIDE_V_TOK + i_hv * V + o_v
    else:
        p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        if STRIDE_BETA_TOK > 0:
            p_beta = beta + bos * STRIDE_BETA_TOK + i_hv * V + o_v
        else:
            p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        if STRIDE_BETA_TOK > 0:
            p_beta = beta + bos * STRIDE_BETA_TOK + i_hv
        else:
            p_beta = beta + bos * HV + i_hv
'''),
 ('''        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)
''',
  '''        if STRIDE_Q_TOK > 0:  # patch 0021
            p_q += STRIDE_Q_TOK
        else:
            p_q += H * K
        if STRIDE_K_TOK > 0:
            p_k += STRIDE_K_TOK
        else:
            p_k += H * K
        p_o += HV * V
        if STRIDE_V_TOK > 0:
            p_v += STRIDE_V_TOK
        else:
            p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        if STRIDE_BETA_TOK > 0:
            p_beta += STRIDE_BETA_TOK
        else:
            p_beta += HV * (V if IS_BETA_HEADWISE else 1)
'''),
]

KDA_HELPERS = '''
# ---- patch 0021: strided q / k / v / beta for the KDA recurrence (VLLM_KDA_STRIDED=1; default = upstream) ----
import os as _p0021_os
_P0021 = _p0021_os.environ.get("VLLM_KDA_STRIDED", "0") == "1"


def _p0021_keep(x):
    """x itself when its layout is [1, tokens, ..., dim] with a dense token payload (then only the distance between
    tokens differs from the dense layout and the kernel takes it as a constexpr); a contiguous copy otherwise."""
    if not _P0021 or x.is_contiguous() or x.dim() < 3 or x.size(0) != 1:
        return x.contiguous()
    inner = 1
    for d in range(x.dim() - 1, 1, -1):
        if x.size(d) != 1 and x.stride(d) != inner:
            return x.contiguous()
        inner *= x.size(d)
    if x.stride(1) < inner:
        return x.contiguous()
    return x


def _p0021_tok_stride(x):
    return 0 if x.is_contiguous() else x.stride(1)
# ---- end patch 0021 helpers ----

'''

KDA_REPL = [
 ('''        COMPUTE_GATE=compute_gate,
        SAFE_GATE=True,
        LOWER_BOUND=lower_bound if lower_bound is not None else -5.0,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o, final_state
''',
  '''        COMPUTE_GATE=compute_gate,
        SAFE_GATE=True,
        LOWER_BOUND=lower_bound if lower_bound is not None else -5.0,
        STRIDE_Q_TOK=_p0021_tok_stride(q),  # patch 0021: 0 for a dense layout
        STRIDE_K_TOK=_p0021_tok_stride(k),
        STRIDE_V_TOK=_p0021_tok_stride(v),
        STRIDE_BETA_TOK=_p0021_tok_stride(beta),
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o, final_state
'''),
 ('''    o, final_state = fused_recurrent_kda_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
''',
  '''    o, final_state = fused_recurrent_kda_fwd(
        q=_p0021_keep(q),  # patch 0021: no copy when only the token stride differs
        k=_p0021_keep(k),
        v=_p0021_keep(v),
        g=g.contiguous(),
        beta=_p0021_keep(beta),
'''),
]
KDA_ANCHOR = "def fused_recurrent_kda_fwd("


def patch_text(kernel_src, kda_src):
    """Returns the patched sources (used by the GPU test as well, so the test runs exactly the patched code)."""
    for old, _ in KERNEL_REPL:
        assert kernel_src.count(old) == 1, f"fused_recurrent.py: expected exactly one of:\n{old}"
    for old, _ in KDA_REPL:
        assert kda_src.count(old) == 1, f"kda.py: expected exactly one of:\n{old}"
    assert kda_src.count(KDA_ANCHOR) == 1, "kda.py: fused_recurrent_kda_fwd not found"
    for old, new in KERNEL_REPL:
        kernel_src = kernel_src.replace(old, new)
    for old, new in KDA_REPL:
        kda_src = kda_src.replace(old, new)
    kda_src = kda_src.replace(KDA_ANCHOR, KDA_HELPERS.lstrip("\n") + "\n" + KDA_ANCHOR, 1)
    return kernel_src, kda_src


if __name__ == "__main__":
    ks, ds = open(F_KERNEL).read(), open(F_KDA).read()
    pk, pd = TAG in ks, TAG in ds
    if MODE == "check":
        print("patched" if (pk and pd) else "unpatched" if not (pk or pd) else f"HALF PATCHED (kernel {pk}, kda {pd})")
        sys.exit(0 if pk == pd else 1)
    if MODE == "revert":
        missing = [f for f in (F_KERNEL, F_KDA) if not os.path.exists(f + SUFFIX)]
        if missing: print("no backup for", missing, "; nothing reverted"); sys.exit(1)
        if not (pk and pd): print("not patched; nothing reverted"); sys.exit(1)
        # the backups are whole files: refuse when something else changed these files after this patch went in
        exp = patch_text(open(F_KERNEL + SUFFIX).read(), open(F_KDA + SUFFIX).read())
        if (ks, ds) != exp:
            print("REFUSED: the installed files are not 'backup + patch 0021' (a later patch on top?). Revert that one first."); sys.exit(1)
        for f in (F_KERNEL, F_KDA): shutil.copy2(f + SUFFIX, f)
        print("reverted both files from *" + SUFFIX); sys.exit(0)
    if MODE == "apply":
        if pk and pd: print("already patched"); sys.exit(0)
        assert not (pk or pd), "half patched: revert first"
        nk, nd = patch_text(ks, ds)  # all asserts before anything is written
        for f in (F_KERNEL, F_KDA): shutil.copy2(f, f + SUFFIX)  # always a fresh backup of what is installed NOW
        open(F_KERNEL, "w").write(nk); open(F_KDA, "w").write(nd)
        for f in (F_KERNEL, F_KDA): py_compile.compile(f, doraise=True)
        print("applied; backups *" + SUFFIX); sys.exit(0)
    print("usage: check|apply|revert"); sys.exit(2)
