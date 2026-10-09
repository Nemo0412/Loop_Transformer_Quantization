"""Weight-only per-group round-to-nearest quantization.

Linear weights are simulated by replacing each weight matrix with its
dequantized floating-point values. No integer GEMM kernel is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def quantize_weight(weight: torch.Tensor, bits: int, group_size: int, scheme: str) -> tuple[torch.Tensor, dict]:
    """Asymmetric or symmetric per-group RTN along the input-feature axis.

    The final group is shorter when in_features is not divisible by group_size.
    A constant group is left unchanged so a zero scale cannot amplify noise.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got {tuple(weight.shape)}")
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")
    if scheme not in ("asymmetric", "symmetric"):
        raise ValueError(scheme)

    out_f, in_f = weight.shape
    qmax = (1 << bits) - 1
    dequant = torch.empty_like(weight)
    n_groups = 0

    for start in range(0, in_f, group_size):
        block = weight[:, start : start + group_size]
        w_min = block.amin(dim=-1, keepdim=True)
        w_max = block.amax(dim=-1, keepdim=True)
        constant = w_min == w_max

        if scheme == "asymmetric":
            scale = (w_max - w_min) / qmax
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            zero = torch.round(-w_min / scale).clamp(0, qmax)
            q = torch.round(block / scale + zero).clamp(0, qmax)
            restored = (q - zero) * scale
        else:
            qmax_s = (1 << (bits - 1)) - 1
            scale = block.abs().amax(dim=-1, keepdim=True) / qmax_s
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            q = torch.round(block / scale).clamp(-qmax_s, qmax_s)
            restored = q * scale

        restored = torch.where(constant, block, restored)
        dequant[:, start : start + group_size] = restored
        n_groups += out_f

    info = {
        "shape": [out_f, in_f],
        "n_params": out_f * in_f,
        "n_groups": n_groups,
        "bits": bits,
        "group_size": group_size,
        "scheme": scheme,
        "packed_weight_bytes": (out_f * in_f * bits) / 8.0,
        "scale_zp_bytes": float(n_groups * (2 + (1 if scheme == "asymmetric" else 0))),
    }
    info["storage_bytes"] = info["packed_weight_bytes"] + info["scale_zp_bytes"]
    return dequant, info


def quantize_linear_weights_(model: nn.Module, bits: int, group_size: int, scheme: str) -> list[dict]:
    """Quantize every nn.Linear in place. Embeddings, RMS scales, gain, and tag stay FP."""
    infos = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        dequant, info = quantize_weight(module.weight.data, bits, group_size, scheme)
        module.weight.data.copy_(dequant)
        if name.startswith("prelude."):
            region = "prelude"
        elif name.startswith("core."):
            region = "core"
        elif name.startswith("coda."):
            region = "coda"
        else:
            region = "other"
        info["name"] = name
        info["region"] = region
        infos.append(info)
    if not infos:
        raise RuntimeError("no Linear modules were quantized")
    return infos


def storage_summary(infos: list[dict]) -> dict:
    def _sum(region: str | None, key: str) -> float:
        return float(sum(i[key] for i in infos if region is None or i["region"] == region))

    fp32_bytes = _sum(None, "n_params") * 4.0
    return {
        "n_quant_params": int(_sum(None, "n_params")),
        "n_core_params": int(_sum("core", "n_params")),
        "n_groups": int(_sum(None, "n_groups")),
        "packed_weight_bytes": _sum(None, "packed_weight_bytes"),
        "scale_zp_bytes": _sum(None, "scale_zp_bytes"),
        "storage_bytes": _sum(None, "storage_bytes"),
        "core_storage_bytes": _sum("core", "storage_bytes"),
        "fp32_linear_bytes": fp32_bytes,
        "low_bit_kernel_used": False,
        "quant_simulation": "fake_quant_dequantized_fp32_matmul",
    }
