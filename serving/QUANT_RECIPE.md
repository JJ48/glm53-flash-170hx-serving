# Weight-quantization recipe

The served model is **weight-only** quantized with [compressed-tensors](https://github.com/neuralmagic/compressed-tensors):

| component | scheme |
|---|---|
| Attention projections (KDA in/out, MLA q_b/kv_b/o) | **INT8**, per-channel |
| MoE experts | **INT4**, group size 32 |
| Dense Linear (catch-all) | INT4, group size 128 |
| lm_head | INT4, group size 32 |
| embeddings, norms, router gates, conv1d, indexer, drafter-adjacent heads | kept BF16 (ignore list) |

Activations are not quantized (W8A16 / W4A16). At serve time this dispatches through vLLM's Marlin
kernels.

## Easiest path: download the published quant

The quantized checkpoint is published on Hugging Face — pull it directly instead of quantizing
yourself:

```
# Published by JKeys LLC: https://huggingface.co/JJ48-24/GLM-5.3-Flash-AWQ-W4A16-aggr-w8-int4g32-mixA
huggingface-cli download JJ48-24/GLM-5.3-Flash-AWQ-W4A16-aggr-w8-int4g32-mixA --local-dir ./glm5.3-flash-w8w4
```

> The **drafter** is not on Hugging Face and is not redistributable (CC BY-NC-ND, internal-only).
> Supply your own compatible speculative drafter — see [`VARIABLE_ACCEPTANCE.md`](VARIABLE_ACCEPTANCE.md) and
> the root README.

## Reproducing the quant from the base model

For transparency, the tool that produces the quant is [`quant/requant.py`](../quant/requant.py). It
adds weight-only INT8/INT4 to the BF16 remainder of a compressed-tensors checkpoint, streaming shard
by shard on CPU, using compressed-tensors' own quantize/pack routines so vLLM's loader sees exactly
the format it expects.

```bash
python quant/requant.py selftest                       # verify pack/unpack round-trips
python quant/requant.py plan --src /path/to/base        # print what each profile would touch
python quant/requant.py run  --src /path/to/base --dst /path/to/out --profile aggr --bits 8 --shard-gb 8
```

**Profiles** (what gets quantized; everything else is copied unchanged):

| profile | targets |
|---|---|
| `kda` | KDA input projections (q/k/v/b/f_a/g_a) + o_proj, dense MLP layers 0–2, shared experts |
| `kda_mla` | `kda` + the MLA layers' q_b_proj / kv_b_proj / o_proj |
| `aggr` | `kda_mla` + lm_head |

An optional GPTQ (calibrated) path is available if you supply a `calib/gptq.py` and per-layer
Hessians; the default is RTN (round-to-nearest), which needs no calibration data. Two of the
profiles assume small model-side patches so the loader keeps `quant_config` on the fused KDA/MLA
projections: they ship as [`kernels/quant_config_patch.py`](../kernels/quant_config_patch.py) (patches 0001/0002) and
are required to load the published checkpoint in the club image; `kernels/apply_stack.sh` applies them.

The exact scheme above (INT8 attention + INT4 experts) is the "aggr" result plus a follow-up pass
that moves shared-experts / dense-MLP / lm_head to INT4 g32; the published HF checkpoint is the final
artifact.
