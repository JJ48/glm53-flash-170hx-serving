#!/usr/bin/env python3
"""Fork patches 0001 + 0002: keep the checkpoint's quantization config on the MLA and KDA attention projections.

The fork builds the MLA projections with quant_config=None ("MLA projections are BF16 in checkpoint") and the KDA layer
stores the temporary None its parent saw on self.quant_config, so every profile that quantizes attention (the published
INT8-attention checkpoint included) fails to load without these two edits:
  0001 models/glm5next/nvidia/model.py  MLA q_b_proj / kv_b_proj / o_proj take the per-module scheme from quant_config
  0002 models/glm5next/nvidia/kda.py    Glm5NextLinearAttention restores self.quant_config after the parent's __init__
The edits are byte-for-byte the ones the production image carries.
Usage: quant_config_patch.py check|apply|revert   (backups: *.orig)"""
import os, shutil, sys
V = os.environ.get("VLLM_DIR", "/usr/local/lib/python3.12/dist-packages/vllm")
M = f"{V}/models/glm5next/nvidia/model.py"
K = f"{V}/models/glm5next/nvidia/kda.py"
M_OLD = "quant_config=None,  # MLA projections are BF16 in checkpoint"
M_NEW = "quant_config=quant_config,  # patched: per-module scheme from the checkpoint quantization_config"
K_MARK = "        vllm_config.quant_config = saved_quant_config\n"
K_ADD = "        self.quant_config = saved_quant_config  # patched: parent captured the temporary None\n"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
m, k = open(M).read(), open(K).read()
if MODE == "check":
    print(f"model(0001):{'patched' if M_NEW in m and M_OLD not in m else 'unpatched'} kda(0002):{'patched' if K_ADD in k else 'unpatched'}")
elif MODE == "apply":
    for p in (M, K):
        if not os.path.exists(p + ".orig"): shutil.copy2(p, p + ".orig")
    if M_OLD in m: open(M, "w").write(m.replace(M_OLD, M_NEW)); print("0001 model.py: applied")
    else: print("0001 model.py: already applied / pattern not found")
    if K_ADD in k: print("0002 kda.py: already applied")
    elif K_MARK in k: open(K, "w").write(k.replace(K_MARK, K_MARK + K_ADD, 1)); print("0002 kda.py: applied")
    else: sys.exit("0002 kda.py: pattern not found")
elif MODE == "revert":
    for p in (M, K):
        if os.path.exists(p + ".orig"): shutil.copy2(p + ".orig", p); print("reverted", os.path.basename(p))
else:
    sys.exit("usage: check|apply|revert")
