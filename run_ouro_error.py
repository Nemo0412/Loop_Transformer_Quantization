#!/usr/bin/env python3
"""Quantization-error-vs-loop study on the real Ouro-1.4B looped LM.

For the frozen Ouro-1.4B base model we weight-quantize the shared looped core
(the 24 decoder layers' q/k/v/o/gate/up/down projections) with per-group RTN at
several bit widths, then run the Universal-Transformer loop and compare, at every
loop t = 1..T, the quantized hidden state against the full-precision (FP) hidden
state.

Because the loop is purely sequential (state at loop t depends only on loops
0..t), a single forward with total_ut_steps = T yields the hidden state at every
intermediate loop count, so one pass covers T = 1..T.

Metrics per (bits, loop):
  err_rms       sqrt(mean (h_q - h_fp)^2)                      absolute error
  rel_err       mean_tok ||h_q - h_fp|| / ||h_fp||             relative error
  cosine        mean_tok cos(h_q, h_fp)
  ce_fp, ce_q   next-token cross-entropy of lm_head(h_t)       downstream effect
  ppl_fp, ppl_q exp(CE)
  logits_mse    mean (lm_head(h_q) - lm_head(h_fp))^2

Weight-only fake quant: weights are dequantized to FP and a normal FP matmul is
used. No low-bit kernel, no speed claim. Embeddings, RMSNorm scales, lm_head and
the exit gate stay FP so the measured error comes purely from the looped core.

Outputs land in ouro_results/ and never touch earlier results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from quantize import quantize_weight  # per-group RTN, dequantized FP

MODEL_ID = "ByteDance/Ouro-1.4B"
SEED = 42
GROUP_SIZE = 128
SCHEME = "asymmetric"
MAX_LOOPS = 8
FP8_E4M3_MAX = 448.0

# Standard weight-only quantization grid vs an FP32 baseline.
QUANT_CONFIGS = {
    "INT8": dict(kind="int", bits=8, scheme="asymmetric"),
    "INT4": dict(kind="int", bits=4, scheme="asymmetric"),
    "FP8": dict(kind="fp8", fmt="e4m3"),
}
DEFAULT_CONFIGS = ("INT8", "INT4", "FP8")
# the shared looped core: these Linear sub-names inside each decoder layer
CORE_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", type=str, nargs="+", default=list(DEFAULT_CONFIGS),
                   choices=list(QUANT_CONFIGS.keys()))
    p.add_argument("--loops", type=int, default=MAX_LOOPS, help="max UT steps")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--eval-seqs", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--group-size", type=int, default=GROUP_SIZE)
    p.add_argument("--scheme", type=str, default=SCHEME)
    p.add_argument("--quant-lm-head", action="store_true",
                   help="also quantize the (non-looped) lm_head")
    p.add_argument("--decompose", action="store_true",
                   help="measure e_{t+1} = p_t + q_t (propagation vs injection)")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def load_ouro(device, loops, dtype=torch.float32):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    for k, v in (("pad_token_id", 0), ("bos_token_id", 0), ("eos_token_id", 0)):
        if not hasattr(cfg, k) or getattr(cfg, k) is None:
            setattr(cfg, k, v)
    # run >4 loops: the KV-cache ceiling is derived from config.total_ut_steps,
    # but we use use_cache=False so this just controls the loop count.
    cfg.total_ut_steps = loops
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=cfg, trust_remote_code=True, torch_dtype=dtype
    )
    model.config.total_ut_steps = loops
    model.model.total_ut_steps = loops
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device), tok


def core_linears(model):
    """Yield (name, module) for every looped-core Linear."""
    for name, m in model.model.layers.named_modules():
        if isinstance(m, torch.nn.Linear) and name.split(".")[-1] in CORE_SUFFIXES:
            yield name, m


def quant_targets(model, quant_lm_head):
    targets = list(core_linears(model))
    if quant_lm_head:
        targets.append(("lm_head", model.lm_head))
    return targets


def stash_weights(targets):
    return [m.weight.data.detach().clone() for _, m in targets]


def restore_weights(targets, stash):
    for (_, m), w in zip(targets, stash):
        m.weight.data.copy_(w)


def quantize_weight_fp8(weight, group_size):
    """Per-group FP8 E4M3 fake quant: scale each group so amax maps to 448,
    cast to float8_e4m3fn, cast back, unscale. Dequantized FP matmul, no kernel."""
    out_f, in_f = weight.shape
    w = weight.float()
    dequant = torch.empty_like(w)
    n_groups = 0
    for start in range(0, in_f, group_size):
        block = w[:, start:start + group_size]
        amax = block.abs().amax(dim=-1, keepdim=True)
        scale = amax / FP8_E4M3_MAX
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = (block / scale).to(torch.float8_e4m3fn).to(torch.float32)
        dequant[:, start:start + group_size] = q * scale
        n_groups += out_f
    info = {
        "shape": [out_f, in_f],
        "n_params": out_f * in_f,
        "n_groups": n_groups,
        "format": "fp8_e4m3",
        "bits": 8,
        "group_size": group_size,
        "packed_weight_bytes": float(out_f * in_f),           # 8 bits = 1 byte
        "scale_zp_bytes": float(n_groups * 2),                # fp16 scale per group
    }
    info["storage_bytes"] = info["packed_weight_bytes"] + info["scale_zp_bytes"]
    return dequant.to(weight.dtype), info


def quantize_weight_cfg(weight, cfg, group_size):
    if cfg["kind"] == "int":
        return quantize_weight(weight, cfg["bits"], group_size, cfg["scheme"])
    if cfg["kind"] == "fp8":
        return quantize_weight_fp8(weight, group_size)
    raise ValueError(cfg)


def quantize_in_place(targets, cfg, group_size):
    infos = []
    for name, m in targets:
        dequant, info = quantize_weight_cfg(m.weight.data, cfg, group_size)
        m.weight.data.copy_(dequant)
        info["name"] = name
        infos.append(info)
    return infos


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_eval_tokens(tok, seq_len, n_seqs, device):
    import pyarrow.parquet as pq

    path = ROOT / "data" / "wikitext" / "wikitext-2-raw-v1-test.parquet"
    table = pq.read_table(path, columns=["text"])
    text = "\n\n".join(table.column("text").to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = ids.numel() // seq_len
    if n < 1:
        raise RuntimeError("not enough tokens")
    windows = ids[: n * seq_len].view(n, seq_len)
    if n_seqs:
        windows = windows[:n_seqs]
    return windows


def iter_batches(tokens, batch_size, device):
    for start in range(0, tokens.shape[0], batch_size):
        yield tokens[start:start + batch_size].to(device)


# ---------------------------------------------------------------------------
# forward helpers
# ---------------------------------------------------------------------------

def loop_hidden_states(model, ids):
    """Return list of per-loop hidden states [T] each (B, S, D)."""
    out = model.model(input_ids=ids, use_cache=False)
    if isinstance(out, (tuple, list)) and len(out) == 3:
        _, hs_list, _ = out
    else:
        raise RuntimeError(f"unexpected OuroModel return: {type(out)}")
    return hs_list


def make_loop_ctx(model, ids):
    """Everything one UT loop needs for a full-sequence (cache-free) forward."""
    from transformers.masking_utils import create_causal_mask

    embed = model.model.embed_tokens(ids)
    cache_position = torch.arange(ids.shape[1], device=ids.device)
    position_ids = cache_position.unsqueeze(0)
    pos_emb = model.model.rotary_emb(embed, position_ids)
    mask = create_causal_mask(
        config=model.config,
        input_embeds=embed,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )
    return dict(embed=embed, pos_emb=pos_emb, mask=mask,
                position_ids=position_ids, cache_position=cache_position)


def one_loop(model, h, ctx):
    """Apply the shared core once (24 layers + final norm) to an arbitrary state."""
    for layer in model.model.layers:
        h = layer(
            h,
            attention_mask=ctx["mask"],
            position_ids=ctx["position_ids"],
            past_key_value=None,
            use_cache=False,
            cache_position=ctx["cache_position"],
            position_embeddings=ctx["pos_emb"],
        )
    return model.model.norm(h)


def _norms(x):
    """RMS over all elements and mean per-token L2 norm."""
    flat = x.reshape(-1, x.shape[-1])
    return (
        math.sqrt(float(x.pow(2).mean().item())),
        float(flat.norm(dim=-1).mean().item()),
    )


def decompose_config(model, targets, stash, label, cfg, args, eval_tokens, device):
    """Measure e_{t+1} = p_t + q_t per loop: propagation vs fresh injection.

    p_t = F(h_q_t; Wq) - F(h_fp_t; Wq)   (last loop's error carried through)
    q_t = F(h_fp_t; Wq) - F(h_fp_t; W)   (this loop's fresh quantization error)
    gain_t = ||p_t|| / ||e_t||           (empirical contraction factor of J)
    """
    T = args.loops
    acc = [dict(e=0.0, p=0.0, q=0.0, en=0, pq=0.0, pqn=0,
                gain=0.0, gainn=0, eprev=0.0, consist=0.0) for _ in range(T)]
    nb = 0

    with torch.inference_mode():
        for ids in iter_batches(eval_tokens, args.batch_size, device):
            ctx = make_loop_ctx(model, ids)
            # FP trajectory
            restore_weights(targets, stash)
            h_fp = []
            h = ctx["embed"]
            for _ in range(T):
                h = one_loop(model, h, ctx)
                h_fp.append(h)
            # quantized trajectory
            quantize_in_place(targets, cfg, args.group_size)
            h_q = []
            h = ctx["embed"]
            for _ in range(T):
                h = one_loop(model, h, ctx)
                h_q.append(h)
            # decomposition (quantized weights still live)
            nb += 1
            for t in range(T):
                e_t = (h_q[t] - h_fp[t]).float()
                if t == 0:
                    # both trajectories start at the same embedding: pure injection
                    p_t = torch.zeros_like(e_t)
                    q_t = e_t
                else:
                    m = one_loop(model, h_fp[t - 1], ctx)      # F(h_fp; Wq)
                    p_t = (h_q[t] - m).float()
                    q_t = (m - h_fp[t]).float()
                a = acc[t]
                e_rms, _ = _norms(e_t)
                p_rms, _ = _norms(p_t)
                q_rms, _ = _norms(q_t)
                a["e"] += e_rms
                a["p"] += p_rms
                a["q"] += q_rms
                a["en"] += 1
                # exactness of the split
                a["consist"] += float((e_t - p_t - q_t).abs().max().item())
                pf = p_t.reshape(-1, p_t.shape[-1])
                qf = q_t.reshape(-1, q_t.shape[-1])
                if t > 0:
                    c = (pf * qf).sum(-1) / (
                        pf.norm(dim=-1).clamp(min=1e-8) * qf.norm(dim=-1).clamp(min=1e-8)
                    )
                    a["pq"] += float(c.mean().item())
                    a["pqn"] += 1
                    prev_rms, _ = _norms((h_q[t - 1] - h_fp[t - 1]).float())
                    if prev_rms > 0:
                        a["gain"] += p_rms / prev_rms
                        a["gainn"] += 1
                    a["eprev"] += prev_rms

    rows = []
    for t in range(T):
        a = acc[t]
        n = max(a["en"], 1)
        e_rms = a["e"] / n
        p_rms = a["p"] / n
        q_rms = a["q"] / n
        gain = a["gain"] / a["gainn"] if a["gainn"] else float("nan")
        # plateau implied by a linear fixed point e* = q / (1 - gain)
        implied = q_rms / (1 - gain) if a["gainn"] and gain < 1 else float("nan")
        rows.append({
            "model": "ouro_1.4b",
            "quant": label,
            "loop": t + 1,
            "e_rms": e_rms,
            "p_rms": p_rms,
            "q_rms": q_rms,
            "propagation_gain": gain,
            "cos_p_q": a["pq"] / a["pqn"] if a["pqn"] else float("nan"),
            "implied_fixed_point": implied,
            "split_max_abs_residual": a["consist"] / n,
        })
    return rows


def plot_decomp(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = _labels(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for lab in labels:
        s = sorted([r for r in rows if r["quant"] == lab], key=lambda r: r["loop"])
        xs = [r["loop"] for r in s]
        axes[0].plot(xs, [r["e_rms"] for r in s], marker="o", label=f"{lab} ||e||")
        axes[0].plot(xs, [r["q_rms"] for r in s], marker="s", linestyle="--",
                     label=f"{lab} ||q|| (injection)")
        axes[1].plot(xs, [r["propagation_gain"] for r in s], marker="o", label=lab)
        axes[2].plot(xs, [r["cos_p_q"] for r in s], marker="o", label=lab)
    axes[0].set_title("total error vs fresh injection")
    axes[0].set_ylabel("RMS")
    axes[1].set_title("propagation gain  ||p_t|| / ||e_{t-1}||")
    axes[1].axhline(1.0, color="r", lw=0.8, linestyle=":")
    axes[1].set_ylabel("gain (rho(J) estimate)")
    axes[2].set_title("cos(p, q)")
    axes[2].axhline(0.0, color="k", lw=0.6)
    for ax in axes:
        ax.set_xlabel("loop (UT step)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Ouro-1.4B error decomposition  e_{t+1} = p_t + q_t")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def loop_ce_sum(model, h, ids):
    """Next-token CE sum and token count for lm_head(h)."""
    logits = model.lm_head(h)
    logits = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
    targets = ids[:, 1:].reshape(-1)
    ce = F.cross_entropy(logits, targets, reduction="sum")
    return float(ce.item()), int(targets.numel())


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def eval_config(model, targets, stash, label, cfg, args, eval_tokens, device):
    """Run FP + quantized rollouts; return per-loop error rows and storage info."""
    T = args.loops
    # accumulators per loop
    acc = [dict(sq=0.0, n=0, rel=0.0, reln=0, cos=0.0, cosn=0,
                lmse=0.0, lmse_n=0,
                ce_fp=0.0, ce_q=0.0, cen=0) for _ in range(T)]

    # 1) quantize
    restore_weights(targets, stash)
    infos = quantize_in_place(targets, cfg, args.group_size)

    with torch.inference_mode():
        for ids in iter_batches(eval_tokens, args.batch_size, device):
            # quantized hidden states
            hs_q = loop_hidden_states(model, ids)
            # FP hidden states (restore originals for this batch, then re-quantize)
            restore_weights(targets, stash)
            hs_fp = loop_hidden_states(model, ids)
            # per-loop logits/CE for FP while FP weights are live
            for t in range(T):
                ce_fp, n_fp = loop_ce_sum(model, hs_fp[t], ids)
                acc[t]["ce_fp"] += ce_fp
            # re-apply quant for the CE on quantized states + next batch
            quantize_in_place(targets, cfg, args.group_size)
            for t in range(T):
                ce_q, n_q = loop_ce_sum(model, hs_q[t], ids)
                acc[t]["ce_q"] += ce_q
                acc[t]["cen"] += n_q
                hq = hs_q[t].float()
                hf = hs_fp[t].float()
                diff = hq - hf
                a = acc[t]
                a["sq"] += float(diff.pow(2).sum().item())
                a["n"] += diff.numel()
                dn = diff.reshape(-1, diff.shape[-1]).norm(dim=-1)
                fn = hf.reshape(-1, hf.shape[-1]).norm(dim=-1).clamp(min=1e-8)
                a["rel"] += float((dn / fn).sum().item())
                a["reln"] += int(dn.numel())
                qn = hq.reshape(-1, hq.shape[-1])
                fnorm = hf.reshape(-1, hf.shape[-1])
                cos = (qn * fnorm).sum(-1) / (
                    qn.norm(dim=-1).clamp(min=1e-8) * fnorm.norm(dim=-1).clamp(min=1e-8)
                )
                a["cos"] += float(cos.sum().item())
                a["cosn"] += int(cos.numel())
                # logits mse (chunked implicitly by batch)
                lq = model.lm_head(hs_q[t]).float()
                lf = model.lm_head(hs_fp[t]).float()
                a["lmse"] += float((lq - lf).pow(2).sum().item())
                a["lmse_n"] += lq.numel()

    rows = []
    for t in range(T):
        a = acc[t]
        err_rms = math.sqrt(a["sq"] / a["n"]) if a["n"] else float("nan")
        ce_fp = a["ce_fp"] / a["cen"] if a["cen"] else float("nan")
        ce_q = a["ce_q"] / a["cen"] if a["cen"] else float("nan")
        rows.append({
            "model": "ouro_1.4b",
            "quant": label,
            "kind": cfg["kind"],
            "bits": cfg.get("bits", 8),
            "group_size": args.group_size,
            "scheme": cfg.get("scheme", cfg.get("fmt", "")),
            "loop": t + 1,
            "err_rms": err_rms,
            "rel_err": a["rel"] / a["reln"] if a["reln"] else float("nan"),
            "cosine": a["cos"] / a["cosn"] if a["cosn"] else float("nan"),
            "logits_mse": a["lmse"] / a["lmse_n"] if a["lmse_n"] else float("nan"),
            "ce_fp": ce_fp,
            "ce_q": ce_q,
            "ppl_fp": math.exp(ce_fp) if ce_fp < 40 else float("inf"),
            "ppl_q": math.exp(ce_q) if ce_q < 40 else float("inf"),
            "ce_gap": ce_q - ce_fp,
        })
    storage = {
        "n_quant_linears": len(infos),
        "n_quant_params": int(sum(i["n_params"] for i in infos)),
        "packed_weight_bytes": float(sum(i["packed_weight_bytes"] for i in infos)),
        "scale_zp_bytes": float(sum(i["scale_zp_bytes"] for i in infos)),
        "fp32_linear_bytes": float(sum(i["n_params"] for i in infos) * 4),
        "low_bit_kernel_used": False,
    }
    return rows, storage


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for k in fields:
                v = r.get(k, "")
                if isinstance(v, float):
                    out[k] = "" if math.isnan(v) else f"{v:.8g}"
                else:
                    out[k] = v
            w.writerow(out)


def _labels(rows):
    order = [c for c in DEFAULT_CONFIGS if any(r["quant"] == c for r in rows)]
    for r in rows:
        if r["quant"] not in order:
            order.append(r["quant"])
    return order


def plot_error(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = _labels(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for lab in labels:
        s = sorted([r for r in rows if r["quant"] == lab], key=lambda r: r["loop"])
        xs = [r["loop"] for r in s]
        axes[0].plot(xs, [r["err_rms"] for r in s], marker="o", label=lab)
        axes[1].plot(xs, [r["rel_err"] for r in s], marker="o", label=lab)
        axes[2].plot(xs, [r["ppl_q"] for r in s], marker="o", label=lab)
    # FP32 perplexity reference
    s0 = sorted([r for r in rows if r["quant"] == labels[0]], key=lambda r: r["loop"])
    axes[2].plot([r["loop"] for r in s0], [r["ppl_fp"] for r in s0], "k--", label="FP32")
    axes[0].set_title("hidden-state error RMS vs loop")
    axes[0].set_ylabel("RMS ||h_q - h_fp||")
    axes[1].set_title("relative error vs loop")
    axes[1].set_ylabel("mean ||Δ|| / ||h_fp||")
    axes[2].set_title("perplexity vs loop")
    axes[2].set_ylabel("WikiText-2 PPL")
    for ax in axes:
        ax.set_xlabel("loop (UT step)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Ouro-1.4B: weight quantization error across loops (FP32 baseline)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(path, rows, storage, meta):
    labels = _labels(rows)
    lines = [
        "# Ouro-1.4B weight-quantization error across loops",
        "",
        f"Frozen `{MODEL_ID}` (hidden 2048, 24 layers, {meta['loops']} UT loops). "
        f"Eval = WikiText-2 raw test, {meta['n_eval_seq']} sequences of {meta['seq_len']} tokens. "
        f"Weight-only per-group quantization (group {meta['group_size']}) on the shared looped "
        "core (q/k/v/o/gate/up/down of all 24 decoder layers). "
        "Baseline = FP32. INT = asymmetric RTN, FP8 = E4M3 per-group scaling. "
        "Dequantized FP32 matmul, no low-bit kernel.",
        "",
        f"Quantized Linears: {storage['n_quant_linears']}, "
        f"params {storage['n_quant_params']:,}.",
        "",
        "## Error and perplexity at the final loop",
        "",
        "| quant | loop | err_rms | rel_err | cosine | PPL (q) | PPL (fp32) | CE gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    T = meta["loops"]
    for lab in labels:
        r = next((x for x in rows if x["quant"] == lab and x["loop"] == T), None)
        if r:
            lines.append(
                f"| {lab} | {r['loop']} | {r['err_rms']:.4f} | {r['rel_err']:.4f} | "
                f"{r['cosine']:.4f} | {r['ppl_q']:.2f} | {r['ppl_fp']:.2f} | {r['ce_gap']:.4f} |"
            )
    lines += ["", "## Error growth across loops", ""]
    for lab in labels:
        s = sorted([r for r in rows if r["quant"] == lab], key=lambda r: r["loop"])
        e1 = s[0]["err_rms"]
        eT = s[-1]["err_rms"]
        growth = eT / e1 if e1 else float("nan")
        lines.append(
            f"- {lab}: err_rms loop1={e1:.4f} -> loop{T}={eT:.4f} "
            f"(x{growth:.2f}); PPL loop{T} quant={s[-1]['ppl_q']:.2f} vs FP32={s[-1]['ppl_fp']:.2f}."
        )
    path.write_text("\n".join(lines))


def run():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    if args.smoke:
        args.loops = 4
        args.configs = ["INT4", "FP8"]
        args.eval_seqs = 2
        args.seq_len = 256
        args.batch_size = 1

    out_dir = ROOT / ("ouro_smoke" if args.smoke else "ouro_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {MODEL_ID} (loops={args.loops}) ...", flush=True)
    model, tok = load_ouro(device, args.loops)
    print("loaded; building eval tokens ...", flush=True)
    eval_tokens = load_eval_tokens(tok, args.seq_len, args.eval_seqs, device)
    print(f"eval tokens {tuple(eval_tokens.shape)}", flush=True)

    targets = quant_targets(model, args.quant_lm_head)
    print(f"quant targets: {len(targets)} Linears", flush=True)
    stash = stash_weights(targets)

    if args.decompose:
        dec_rows = []
        for label in args.configs:
            cfg = QUANT_CONFIGS[label]
            print(f"=== decompose {label} ===", flush=True)
            rows = decompose_config(model, targets, stash, label, cfg, args,
                                    eval_tokens, device)
            for r in rows:
                print(f"  loop {r['loop']}: |e|={r['e_rms']:.4f} |p|={r['p_rms']:.4f} "
                      f"|q|={r['q_rms']:.4f} gain={r['propagation_gain']:.4f} "
                      f"cos(p,q)={r['cos_p_q']:.4f} "
                      f"implied_e*={r['implied_fixed_point']:.4f} "
                      f"resid={r['split_max_abs_residual']:.2e}", flush=True)
            dec_rows.extend(rows)
            write_csv(out_dir / "ouro_error_decomposition.csv", dec_rows)
        restore_weights(targets, stash)
        plot_decomp(dec_rows, out_dir / "plots" / "ouro_error_decomposition.png")
        print("done (decompose)", flush=True)
        return

    all_rows = []
    storage = {}
    storages = {}
    for label in args.configs:
        cfg = QUANT_CONFIGS[label]
        print(f"=== {label} ===", flush=True)
        rows, storage = eval_config(model, targets, stash, label, cfg, args, eval_tokens, device)
        storages[label] = storage
        for r in rows:
            print(f"  loop {r['loop']}: err_rms={r['err_rms']:.4f} rel={r['rel_err']:.4f} "
                  f"cos={r['cosine']:.4f} ppl_q={r['ppl_q']:.2f} ppl_fp={r['ppl_fp']:.2f}", flush=True)
        all_rows.extend(rows)
        write_csv(out_dir / "ouro_error_vs_loop.csv", all_rows)
    # make sure the model is back to FP
    restore_weights(targets, stash)

    meta = {
        "model": MODEL_ID,
        "loops": args.loops,
        "seq_len": args.seq_len,
        "n_eval_seq": int(eval_tokens.shape[0]),
        "n_eval_tokens": int(eval_tokens.numel()),
        "group_size": args.group_size,
        "baseline": "fp32",
        "configs": args.configs,
        "quant_lm_head": args.quant_lm_head,
        "device": torch.cuda.get_device_name(0),
        "low_bit_kernel_used": False,
    }
    plot_error(all_rows, out_dir / "plots" / "ouro_error_vs_loop.png")
    write_report(out_dir / "report.md", all_rows, storage, meta)
    (out_dir / "summary.json").write_text(json.dumps({**meta, "storage": storages}, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    run()
