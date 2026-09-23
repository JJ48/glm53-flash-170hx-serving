# The patch stack

The results in this repo were measured on the **club-170hx vLLM image**
(`ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905`) with the patches in [`kernels/`](../kernels/) applied
in the order below. Apart from the two loader fixes (0001/0002), which are always on, each patch changes nothing until
its flag is set (0009 acts only with an MTP drafter). The table says which flags production sets.

```bash
# inside the container, as root
bash kernels/apply_stack.sh          # applies everything below, in order; idempotent
```

Apply the **whole sequence** even for patches you leave switched off: later patches were written against the tree with
every earlier patch applied, and some share files with earlier ones. [`launch.example.sh`](launch.example.sh) sets the
production flags.

## Order, flags and effect

| # | file | flag (production value) | what it does | measured effect on this box |
|---|---|---|---|---|
| 0001, 0002 | `quant_config_patch.py` | always on | keep the checkpoint's quantization config on the MLA and KDA attention projections | **required** to load the published INT8-attention checkpoint |
| 0003 | `apply_spec_k.py` + `files/` | `VLLM_SPEC_ROLL=1` | per-request draft length, the scheduler and runner hooks for the rolling-acceptance rule, a draft-confidence gate | enables the rule in [`VARIABLE_ACCEPTANCE.md`](VARIABLE_ACCEPTANCE.md) |
| | `spec_roll.py` | `VLLM_SPEC_ROLL_STEP_MS=...` | the rolling-acceptance rule itself (iteration 4.7), copied over the older copy 0003 installs | see [`VARIABLE_ACCEPTANCE.md`](VARIABLE_ACCEPTANCE.md) |
| 0005 | `dflash2_conf_patch.py` | off (no `VLLM_SPEC_CONF_MIN`) | per-position draft confidence for the DFlash2 drafter | inert in production |
| 0006 | `gate_async_patch.py` | off | confidence gate for the async scheduler | inert in production |
| 0009 | `dynsd_mtp_patch.py` | automatic, MTP only | lets an autoregressive (MTP) drafter coexist with the dynamic-draft schedule | used by the MTP fallback only |
| 0010 | `gate_sync_patch.py` | off | removes a host stall from 0006 | inert in production |
| 0011 | `mla_split_patch.py` | `VLLM_MLA_SPARSE_SPLIT_MODE=fill` | sparse-MLA KV-split heuristic for a 70-SM part | +3.6 % tok/s at 7 drafts, +4.2 % at 4 |
| 0012 | `pp_cadence_patch.py` | off | two pipeline steps in flight for one request | inert in production |
| 0013 | `fla_rec_patch.py` | off (defaults) | launch-config knobs for the linear-attention decode kernel | inert in production |
| 0014 | `glm5_compile_patch.py` | `VLLM_GLM5_COMPILE=1` + `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | torch.compile (inductor fusion) for GLM-5.3-Flash | -0.9 % decode step. **Requires 0038**, see below |
| | `moe_audit_patch.py` | off | debug print of distinct experts per step | inert |
| 0015 | `pp_packed_patch.py` | `VLLM_PP_PACKED_HOP=1` | one NCCL op per pipeline hop | -2.6 % decode step |
| 0016 | `marlin_cfg_patch.py` | off (auto) | Marlin MoE tile knobs | auto was best of 44 configs |
| 0017 | `pp_hop_v2_patch.py` | **must stay off** | further hop cuts | breaks on the first real request and disables 0022 |
| 0018 | `marlin_dev_patch.py` | `VLLM_MARLIN_DEV_SO=<.so>` + `MARLIN_DEV_STAGES=6` | routes the MoE Marlin GEMM to a standalone build with a 6-stage decode pipeline ([`kernels/marlin/`](../kernels/marlin/)) | +2.9 % tok/s (122.1 vs 118.7, English suite) |
| 0021 | `kda_strided_patch.py` | `VLLM_KDA_STRIDED=1` | the KDA recurrence reads q/k/v/beta in place (no copies inside the CUDA graphs) | 0021 + 0022 + 0025 + 0026 together: English suite 116.2 -> 121.8 tok/s, Spec-Bench at 8 users about +43 % |
| 0022 | `pp_draft_event_patch.py` | `VLLM_PP_SPLIT_DRAFT_EVENT=1` | the first pipeline rank prepares the next step while the drafter runs | (with the row above) |
| 0025 | `topp_triton_rows_patch.py` | `VLLM_TOPP_TRITON_MIN_ROWS=2` | one-launch Triton top-p for any number of logit rows | (with the row above) |
| 0026 | `spec_roll_solo_patch.py` | `VLLM_SPEC_ROLL_SOLO=2` | the rolling rule only while at most 2 requests run, so busy batches keep full CUDA graphs | (with the row above) |
| 0035 | `aot_mm_key_patch.py` | `VLLM_AOT_MM_KEY=1` (image input) | the AOT compile-cache key tells an image-enabled boot from a text-only one | image input boots |
| 0036 | `mm_budget_cap_patch.py` | `VLLM_GLM5_MM_BUDGET=1` (image input) | encoder budget sized at the processor's 8,000-token cap | 3:4 photos up to 3000x4000 accepted |
| 0038 | `wp_precompute_fix_patch.py` | **`VLLM_GLM5_WP_FIX=1`** | fixes 0014's indexer head-gate precompute | see below |

The results tables in the README were measured with 0001-0026 and the production flags above, text-only (0035/0036
came later and are inert without their flags).

## Known issue fixed 2026-09-23: patch 0014 and long prompts (patch 0038)

Patch 0014 pre-computed each sparse-attention indexer's fp32 head-gate at the end of the language model's
`load_weights`, but only while it was still unset. The multimodal wrapper calls that `load_weights` twice (the vision
tensors sit in the middle of the checkpoint), so the first call froze the head-gate of every layer whose weights had not
been read yet at zero: 9 of the 11 sparse-attention layers. A zero head-gate makes every indexer score equal, and the
top-k then picks the first 512 pools, so for any prompt longer than 2,048 tokens those layers read only its first 2,048
tokens. Short prompts were unaffected (the model attends densely up to 2,048 tokens), which is why short-prompt
benchmarks, perplexity windows and our exact-answer canary never showed it.

Patch 0038 recomputes the head-gates on every `load_weights` call and logs
`patch 0038: indexer head-gates after this load_weights call: N indexers, Z all-zero`; the last line per rank must read
`0 all-zero`. Measured on the production config, synthetic long documents with planted facts:

| | before (0014 alone) | with 0038 |
|---|---|---|
| retrieval-shaped prompt (~3.8k tokens), 12 questions | 2 correct, 9 wrong | 12 correct |
| whole 30.6k-token document, planted facts copied at prefill | 2/6 | 5/6 |
| decode at 16k-token prompts, 1 user | 65.0 tok/s (acceptance 1.60) | 129.7 tok/s (acceptance 4.00) |
| time to first token, 8k / 16k prompts | 3.14 / 5.58 s | 3.13 / 5.53 s |

**Always set `VLLM_GLM5_WP_FIX=1` together with `VLLM_GLM5_COMPILE=1`.** Without 0014 (compile off) the fork computes
the head-gate lazily after loading and is not affected.

## The standalone Marlin kernel (patch 0018)

`kernels/marlin/build.sh` checks out upstream vLLM at 8e92248f79, applies `marlin_stages.patch` (extra 6- and 8-stage
decode instantiations and the `MARLIN_DEV_STAGES` knob, about 20 lines), adds `bindings_dev.cpp` (registers the GEMM
under its own `_marlin_dev` namespace with the fork's op schema) and builds it with `build_marlin_dev.py` for sm_80.
Point `VLLM_MARLIN_DEV_SO` at the result and set `MARLIN_DEV_STAGES=6`. Outputs are bit-identical to the fork's kernel
for every stage count; without the flag the fork's own kernel runs.

## Verification

`kernels/apply_stack.sh`, run on a fresh club-170hx container (no GPU, no network), reproduces the production server's
vLLM tree **byte for byte**: all 2,431 `.py` files hash identically to production's manifest, captured 2026-09-23 with
patch 0038 in place. (Four patch files had their module docstrings edited to drop internal paths; the edits they make
to vLLM are unchanged, as the hash match shows.)
