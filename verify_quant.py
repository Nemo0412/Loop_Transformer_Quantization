#!/usr/bin/env python3
"""Verify the weight quantizers really produce low-bit weights.

Reads a few real Ouro-1.4B weight matrices straight from the safetensors file
(no full model load, runs on CPU) and checks, per quantization config:
  - the number of distinct values inside a quantization group (must be <= 2^bits)
  - the relative weight error ||W_q - W|| / ||W||
  - the ratio of weight error between configs (expected ~2^(b1-b2) for INT)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from quantize import quantize_weight
from run_ouro_error import QUANT_CONFIGS, quantize_weight_cfg

SNAP = Path(
    "/scratch/ll5914/hf_home/hub/models--ByteDance--Ouro-1.4B/snapshots/"
    "574fa66cb8bf5abdc979642d01cf2b79b16bfab1/model.safetensors"
)
GROUP = 128
NAMES = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.12.mlp.down_proj.weight",
]


def main():
    from safetensors import safe_open

    with safe_open(str(SNAP), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"safetensors has {len(keys)} tensors")
        for name in NAMES:
            if name not in keys:
                print(f"!! missing {name}; sample keys: {keys[:5]}")
                continue
            W = f.get_tensor(name).float()
            wn = W.norm()
            print(f"\n=== {name}  shape={tuple(W.shape)} "
                  f"absmax={float(W.abs().max()):.4f} ===")
            errs = {}
            for label in ("INT8", "INT4", "FP8"):
                Wq, info = quantize_weight_cfg(W, QUANT_CONFIGS[label], GROUP)
                rel = float((Wq - W).norm() / wn)
                errs[label] = rel
                # distinct levels inside one group (row 0, first group)
                grp = Wq[0, :GROUP]
                n_uniq = int(torch.unique(grp).numel())
                # distinct levels across a whole row (many groups -> more levels)
                n_uniq_row = int(torch.unique(Wq[0]).numel())
                changed = float((Wq != W).float().mean())
                print(f"  {label:5s} rel_w_err={rel:.5f}  "
                      f"uniq_in_group={n_uniq:3d}  uniq_in_row={n_uniq_row:5d}  "
                      f"frac_changed={changed:.4f}  bits={info.get('bits')}")
            print(f"  ratio INT4/INT8 weight error = {errs['INT4']/errs['INT8']:.2f} "
                  f"(expect ~16 for 4 vs 8 bit)")
            print(f"  ratio FP8/INT8  weight error = {errs['FP8']/errs['INT8']:.2f}")

        # explicit level-count check on a controlled tensor
        print("\n=== level-count sanity on a single group ===")
        x = torch.randn(1, GROUP)
        for bits in (8, 4, 2):
            xq, _ = quantize_weight(x, bits, GROUP, "asymmetric")
            print(f"  INT{bits}: unique levels in one group = "
                  f"{int(torch.unique(xq).numel())} (max {2**bits})")


if __name__ == "__main__":
    main()
