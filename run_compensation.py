#!/usr/bin/env python3
"""Phase-1 compensation study on the frozen TRM-LM-v2-10M.

Compares inference-time quantization-error compensators that never read a
full-precision hidden state at inference:

  naive        no correction
  mean         loop-wise mean error subtraction (Method A)
  lowrank      loop-aware low-rank map, independent and shared (Method B)
  dynamics     low-rank error dynamics, free-running (Method C)
  oracle       full-state alpha=1 correction before every non-final loop (ceiling)

The mean and low-rank correctors are fit sequentially on the *compensated*
quantized rollout, so the calibration trajectory matches the evaluation
trajectory. Dynamics is fit from the uncompensated rollout with a DAgger refit.
Full-precision states are used only to build calibration targets and to log
prediction quality.

Deliverables land in comp_results/ and never touch the earlier LM/ARC results.
"""

from __future__ import annotations

import argparse
import copy
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

import compensate as C
from data import build_or_load_tokens
from forward import (
    logits_from_state,
    loop_body,
    mse_sum,
    prelude,
    run_core,
    token_ce_sum,
)
from modeling_trm2 import TRM2
from quantize import quantize_linear_weights_, storage_summary

SEED = 42
GROUP_SIZE = 128
SCHEME = "asymmetric"
LOOP_COUNTS = (2, 4, 6, 8)
BITS = (4, 3, 2)
LOWRANK_RANKS = (4, 16, 64)
DYN_RANKS = (16, 64)
BOOTSTRAP = 1000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--calib-seqs", type=int, default=128)
    p.add_argument("--max-eval-seqs", type=int, default=0, help="0 = all eval windows")
    p.add_argument("--bits", type=int, nargs="+", default=list(BITS))
    p.add_argument("--loops", type=int, nargs="+", default=list(LOOP_COUNTS))
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def load_model(path: Path, device) -> TRM2:
    model = TRM2()
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    model.load_state_dict(obj)
    model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def make_quantized(fp: TRM2, bits: int, device):
    quant = copy.deepcopy(fp).cpu()
    infos = quantize_linear_weights_(quant, bits, GROUP_SIZE, SCHEME)
    if len(infos) != 28:
        raise RuntimeError(f"expected 28 Linear layers, found {len(infos)}")
    for p in quant.parameters():
        p.requires_grad_(False)
    return quant.to(device).eval(), storage_summary(infos)


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

def fp_states(model, ids, n_loops):
    e, cos, sin = prelude(model, ids)
    return run_core(model, e, e, n_loops, cos, sin), (cos, sin)


def quant_rollout(model, ids, n_loops, corrector: C.Compensator):
    """Return observable per-loop states (pre-correction) and final logits.

    A compensator sees only the live quantized state. The corrected state is
    carried into the next loop; the returned states are the pre-correction
    observables used for fitting and logging.
    """
    e, cos, sin = prelude(model, ids)
    corrector.reset()
    h = e
    observed = []
    for i in range(n_loops):
        h = loop_body(model, h, e, i, cos, sin)
        observed.append(h)
        if i < n_loops - 1:
            h = corrector.correct(i, h)
    return observed, logits_from_state(model, h, cos, sin)


def oracle_rollout(model, ids, n_loops, fp_s):
    e, cos, sin = prelude(model, ids)
    h = e
    for i in range(n_loops):
        h = loop_body(model, h, e, i, cos, sin)
        if i < n_loops - 1:
            h = fp_s[i]  # alpha=1 full-state replacement
    return logits_from_state(model, h, cos, sin)


def iter_batches(tokens, batch_size, device):
    for start in range(0, tokens.shape[0], batch_size):
        yield tokens[start:start + batch_size].to(device)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def collect_loop_t(model_q, model_fp, tokens, n_loops, t, corrector, batch_size, device, stride):
    """Pre-correction quantized state h_t and true error e_t under `corrector`."""
    hs, es = [], []
    with torch.inference_mode():
        for ids in iter_batches(tokens, batch_size, device):
            obs, _ = quant_rollout(model_q, ids, n_loops, corrector)
            fps, _ = fp_states(model_fp, ids, n_loops)
            h = obs[t][:, ::stride].reshape(-1, obs[t].shape[-1]).float().cpu()
            e = (obs[t] - fps[t])[:, ::stride].reshape(-1, obs[t].shape[-1]).float().cpu()
            hs.append(h)
            es.append(e)
    return torch.cat(hs, 0), torch.cat(es, 0)


def collect_all_errors(model_q, model_fp, tokens, n_loops, batch_size, device, stride):
    """True errors at every loop plus observable loop-0 state, uncompensated."""
    errs = [[] for _ in range(n_loops)]
    h0s = []
    none = C.NoComp()
    with torch.inference_mode():
        for ids in iter_batches(tokens, batch_size, device):
            obs, _ = quant_rollout(model_q, ids, n_loops, none)
            fps, _ = fp_states(model_fp, ids, n_loops)
            h0s.append(obs[0][:, ::stride].reshape(-1, obs[0].shape[-1]).float().cpu())
            for t in range(n_loops):
                errs[t].append((obs[t] - fps[t])[:, ::stride].reshape(-1, obs[t].shape[-1]).float().cpu())
    return [torch.cat(x, 0) for x in errs], torch.cat(h0s, 0)


def fit_mean(model_q, model_fp, tokens, n_loops, batch_size, device, stride, fit_lambda):
    mus, lams = [], []
    comp = C.MeanComp(mus, lams)
    for t in range(n_loops - 1):
        h_t, e_t = collect_loop_t(model_q, model_fp, tokens, n_loops, t, comp, batch_size, device, stride)
        mu = C.fit_mean(h_t, e_t).to(device)
        lam = C.best_lambda(mu.cpu(), e_t) if fit_lambda else 1.0
        mus.append(mu)
        lams.append(lam)
    return C.MeanComp(mus, lams, name=("mean_fitlam" if fit_lambda else "mean"))


def fit_lowrank(model_q, model_fp, tokens, n_loops, rank, shared, batch_size, device, stride):
    factors = []
    comp = C.LowRankComp(factors, shared=shared)
    if shared:
        # fit one map on the uncompensated loop-1 statistics, reuse everywhere
        h0, e0 = collect_loop_t(model_q, model_fp, tokens, n_loops, 0, C.NoComp(), batch_size, device, stride)
        u, v = C.fit_lowrank(h0.to(device), e0.to(device), rank)
        factors.extend([(u, v)] * (n_loops - 1))
        return C.LowRankComp(factors, shared=True, name=f"lowrank_shared_r{rank}")
    for t in range(n_loops - 1):
        h_t, e_t = collect_loop_t(model_q, model_fp, tokens, n_loops, t, comp, batch_size, device, stride)
        u, v = C.fit_lowrank(h_t.to(device), e_t.to(device), rank)
        factors.append((u, v))
    return C.LowRankComp(factors, shared=False, name=f"lowrank_indep_r{rank}")


def fit_dynamics(model_q, model_fp, tokens, n_loops, rank, batch_size, device, stride):
    errs, h0 = collect_all_errors(model_q, model_fp, tokens, n_loops, batch_size, device, stride)
    comp = C.fit_dynamics(errs, h0, rank, device, refine=1)
    comp.name = f"dynamics_r{rank}"
    return comp


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate(model_q, model_fp, tokens, n_loops, corrector, batch_size, device, want_pred):
    nll = n_tok = 0.0
    mse_sq = mse_n = 0.0
    pred_sq = pred_n = cos_sum = cos_n = 0.0
    per_seq_nll, per_seq_n = [], []
    with torch.inference_mode():
        for ids in iter_batches(tokens, batch_size, device):
            obs, logits = quant_rollout(model_q, ids, n_loops, corrector)
            fps, (cos, sin) = fp_states(model_fp, ids, n_loops)
            ref = logits_from_state(model_fp, fps[-1], cos, sin)
            b_nll, b_n = token_ce_sum(logits, ids)
            nll += b_nll
            n_tok += b_n
            sq, m = mse_sum(logits, ref)
            mse_sq += sq
            mse_n += m
            # per-sequence CE for bootstrap
            for s in range(ids.shape[0]):
                s_nll, s_n = token_ce_sum(logits[s:s + 1], ids[s:s + 1])
                per_seq_nll.append(s_nll)
                per_seq_n.append(s_n)
            if want_pred:
                corrector.reset()
                for i in range(n_loops - 1):
                    pred = corrector.predicted_error(i, obs[i])
                    corrector.correct(i, obs[i])  # advance recurrent state
                    true = (obs[i] - fps[i]).float()
                    diff = (pred.float() - true)
                    pred_sq += float(diff.pow(2).sum().item())
                    pred_n += diff.numel()
                    pf = pred.float().reshape(-1, pred.shape[-1])
                    tf = true.reshape(-1, true.shape[-1])
                    tc = (pf * tf).sum(-1) / (pf.norm(dim=-1).clamp(min=1e-8) * tf.norm(dim=-1).clamp(min=1e-8))
                    cos_sum += float(tc.sum().item())
                    cos_n += int(tc.numel())
    ce = nll / n_tok
    out = {
        "ce": ce,
        "ppl": math.exp(ce) if ce < 40 else float("inf"),
        "logits_mse": mse_sq / mse_n if mse_n else 0.0,
        "n_tokens": int(n_tok),
        "pred_mse": pred_sq / pred_n if pred_n else float("nan"),
        "pred_cosine": cos_sum / cos_n if cos_n else float("nan"),
        "per_seq_nll": np.asarray(per_seq_nll),
        "per_seq_n": np.asarray(per_seq_n),
    }
    return out


def bootstrap_ce(per_seq_nll, per_seq_n, seed, reps=BOOTSTRAP):
    rng = np.random.default_rng(seed)
    n = len(per_seq_nll)
    if n == 0:
        return float("nan"), float("nan")
    ces = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, n, n)
        ces[r] = per_seq_nll[idx].sum() / max(per_seq_n[idx].sum(), 1)
    return float(np.percentile(ces, 2.5)), float(np.percentile(ces, 97.5))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def gap_closed(base_ce, new_ce, fp_ce):
    gap = base_ce - fp_ce
    if abs(gap) < 1e-9:
        return float("nan")
    return (base_ce - new_ce) / gap


def run():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    stride = 1
    out_dir = ROOT / ("comp_smoke" if args.smoke else "comp_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    fp = load_model(ROOT / "checkpoints" / "trm2.pt", device)
    eval_tokens, calib_tokens, meta = build_or_load_tokens(
        ROOT, ROOT / "checkpoints" / "tokenizer.json", args.seq_len, args.calib_seqs, SEED
    )
    if args.max_eval_seqs:
        eval_tokens = eval_tokens[: args.max_eval_seqs]
    if args.smoke:
        eval_tokens = eval_tokens[:4]
        calib_tokens = calib_tokens[:4]
        args.bits = [4]
        args.loops = [2]
    print(f"eval {eval_tokens.shape} calib {calib_tokens.shape} sha {meta['eval_sha256'][:12]}", flush=True)

    bits_list = args.bits
    loops_list = args.loops
    quant = {b: make_quantized(fp, b, device) for b in bits_list}

    rows = []
    err_rows = []
    for bits in bits_list:
        model_q, storage = quant[bits]
        for n_loops in loops_list:
            print(f"=== INT{bits} loops={n_loops} ===", flush=True)
            # FP + naive baselines
            fp_eval = evaluate(fp, fp, eval_tokens, n_loops, C.NoComp(), args.batch_size, device, False)
            naive = evaluate(model_q, fp, eval_tokens, n_loops, C.NoComp(), args.batch_size, device, False)
            base_ce = naive["ce"]
            fp_ce = fp_eval["ce"]

            # oracle ceiling
            oracle_nll = oracle_n = 0.0
            with torch.inference_mode():
                for ids in iter_batches(eval_tokens, args.batch_size, device):
                    fps, _ = fp_states(fp, ids, n_loops)
                    logits = oracle_rollout(model_q, ids, n_loops, fps)
                    o_nll, o_n = token_ce_sum(logits, ids)
                    oracle_nll += o_nll
                    oracle_n += o_n
            oracle_ce = oracle_nll / oracle_n

            # error decomposition on the uncompensated rollout
            err_rows.extend(decompose(model_q, fp, eval_tokens, n_loops, bits, args.batch_size, device))

            # build correctors
            correctors = [C.NoComp()]
            correctors.append(fit_mean(model_q, fp, calib_tokens, n_loops, args.batch_size, device, stride, False))
            correctors.append(fit_mean(model_q, fp, calib_tokens, n_loops, args.batch_size, device, stride, True))
            for rank in LOWRANK_RANKS:
                correctors.append(fit_lowrank(model_q, fp, calib_tokens, n_loops, rank, False, args.batch_size, device, stride))
            correctors.append(fit_lowrank(model_q, fp, calib_tokens, n_loops, 16, True, args.batch_size, device, stride))
            for rank in DYN_RANKS:
                correctors.append(fit_dynamics(model_q, fp, calib_tokens, n_loops, rank, args.batch_size, device, stride))

            for comp in correctors:
                want_pred = comp.name not in ("none",)
                res = evaluate(model_q, fp, eval_tokens, n_loops, comp, args.batch_size, device, want_pred)
                lo, hi = bootstrap_ce(res["per_seq_nll"], res["per_seq_n"], SEED)
                frac = (base_ce - res["ce"]) / (base_ce - oracle_ce) if abs(base_ce - oracle_ce) > 1e-9 else float("nan")
                rows.append({
                    "model": "trm_lm_10m",
                    "bits": bits,
                    "loops": n_loops,
                    "method": comp.name,
                    "ce": res["ce"],
                    "ppl": res["ppl"],
                    "ce_ci_lo": lo,
                    "ce_ci_hi": hi,
                    "logits_mse": res["logits_mse"],
                    "fp_ce": fp_ce,
                    "fp_ppl": fp_eval["ppl"],
                    "naive_ce": base_ce,
                    "oracle_ce": oracle_ce,
                    "oracle_ppl": math.exp(oracle_ce) if oracle_ce < 40 else float("inf"),
                    "ce_gap_closed": gap_closed(base_ce, res["ce"], fp_ce),
                    "fraction_of_oracle": frac,
                    "pred_mse": res["pred_mse"],
                    "pred_cosine": res["pred_cosine"],
                    "extra_params": comp.extra_params(),
                    "extra_bytes": comp.extra_params() * 4,
                    "n_eval_tokens": res["n_tokens"],
                    "low_bit_kernel_used": False,
                    "fp_state_at_inference": False,
                })
                print(f"  {comp.name:20s} ce={res['ce']:.4f} ppl={res['ppl']:.3f} "
                      f"gap={rows[-1]['ce_gap_closed']:.3f} frac_oracle={frac:.3f} "
                      f"pred_cos={res['pred_cosine']:.3f}", flush=True)

            write_csv(out_dir / "compensation_results.csv", rows)
            write_csv(out_dir / "error_decomposition.csv", err_rows)

    plot_methods(rows, out_dir / "plots" / "compensation_lm.png")
    plot_decomp(err_rows, out_dir / "plots" / "error_decomposition_lm.png")
    write_report(out_dir / "report.md", rows, err_rows, meta)
    (out_dir / "summary.json").write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "n_eval_tokens": meta["n_eval_tokens"],
        "n_eval_predictions": meta["n_eval_predictions"],
        "bits": bits_list,
        "loops": loops_list,
        "low_bit_kernel_used": False,
        "fp_state_at_inference": False,
    }, indent=2))
    print("done", flush=True)


def decompose(model_q, model_fp, tokens, n_loops, bits, batch_size, device):
    """Measure ||p||, ||q||, ||e||, cos(p,q), cos(e_t,e_{t+1}), gain ||p||/||e||."""
    d = None
    acc = [dict(e=0.0, p=0.0, q=0.0, pq=0.0, en=0, cos=0.0, cosn=0.0, prevn=0) for _ in range(n_loops)]
    prev = {}
    with torch.inference_mode():
        for ids in iter_batches(tokens, batch_size, device):
            e_pre, cos, sin = prelude(model_q, ids)
            e_fp, cosf, sinf = prelude(model_fp, ids)
            qh = e_pre
            fh = e_fp
            local_prev = None
            prev_e = None
            for i in range(n_loops):
                q_next = loop_body(model_q, qh, e_pre, i, cos, sin)
                f_next = loop_body(model_fp, fh, e_fp, i, cosf, sinf)
                q_on_fp = loop_body(model_q, fh, e_fp, i, cos, sin)
                e_t = (q_next - f_next).float()
                q_t = (q_on_fp - f_next).float()      # injection
                p_t = (q_next - q_on_fp).float()      # propagation
                d = e_t.shape[-1]
                a = acc[i]
                a["e"] += float(e_t.pow(2).sum().item())
                a["p"] += float(p_t.pow(2).sum().item())
                a["q"] += float(q_t.pow(2).sum().item())
                a["en"] += e_t.numel()
                pf = p_t.reshape(-1, d)
                qf = q_t.reshape(-1, d)
                pq = (pf * qf).sum(-1) / (pf.norm(dim=-1).clamp(min=1e-8) * qf.norm(dim=-1).clamp(min=1e-8))
                a["pq"] += float(pq.sum().item())
                a["cosn"] += int(pq.numel())
                ef = e_t.reshape(-1, d)
                if prev_e is not None:
                    cc = (ef * prev_e).sum(-1) / (ef.norm(dim=-1).clamp(min=1e-8) * prev_e.norm(dim=-1).clamp(min=1e-8))
                    a["cos"] += float(cc.sum().item())
                    a["prevn"] += int(cc.numel())
                prev_e = ef
                qh, fh = q_next, f_next
    rows = []
    for i, a in enumerate(acc):
        e_rms = math.sqrt(a["e"] / a["en"])
        p_rms = math.sqrt(a["p"] / a["en"])
        q_rms = math.sqrt(a["q"] / a["en"])
        rows.append({
            "model": "trm_lm_10m",
            "bits": bits,
            "loops": n_loops,
            "loop_index": i + 1,
            "e_rms": e_rms,
            "p_rms": p_rms,
            "q_rms": q_rms,
            "propagation_gain": p_rms / e_rms if e_rms else float("nan"),
            "injection_over_e": q_rms / e_rms if e_rms else float("nan"),
            "cos_p_q": a["pq"] / a["cosn"] if a["cosn"] else float("nan"),
            "cos_e_prev": a["cos"] / a["prevn"] if a["prevn"] else float("nan"),
        })
    return rows


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen and not isinstance(row[k], np.ndarray):
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {}
            for k in fields:
                v = row.get(k, "")
                if isinstance(v, float):
                    out[k] = "" if math.isnan(v) else f"{v:.8g}"
                elif isinstance(v, bool):
                    out[k] = "true" if v else "false"
                else:
                    out[k] = v
            w.writerow(out)


def plot_methods(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    bits_set = sorted({r["bits"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(1, len(bits_set), figsize=(4.6 * len(bits_set), 3.8), squeeze=False)
    methods = ["naive", "none", "mean", "lowrank_indep_r16", "dynamics_r16", "oracle"]
    for ax, bits in zip(axes[0], bits_set):
        for name in ["none", "mean", "mean_fitlam", "lowrank_indep_r16", "lowrank_indep_r64", "dynamics_r16", "dynamics_r64"]:
            pts = sorted([r for r in rows if r["bits"] == bits and r["method"] == name], key=lambda r: r["loops"])
            if not pts:
                continue
            ax.plot([p["loops"] for p in pts], [p["ppl"] for p in pts], marker="o", label=name)
        fp_pts = sorted([r for r in rows if r["bits"] == bits and r["method"] == "none"], key=lambda r: r["loops"])
        if fp_pts:
            ax.plot([p["loops"] for p in fp_pts], [p["fp_ppl"] for p in fp_pts], "k--", label="fp")
            ax.plot([p["loops"] for p in fp_pts], [p["oracle_ppl"] for p in fp_pts], color="gray", linestyle=":", label="oracle")
        ax.set_title(f"INT{bits}")
        ax.set_xlabel("loops")
        ax.set_ylabel("perplexity")
        ax.grid(True, alpha=0.3)
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("TRM-LM: perplexity by compensator")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_decomp(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    bits_set = sorted({r["bits"] for r in rows}, reverse=True)
    max_loops = max(r["loops"] for r in rows)
    sel = [r for r in rows if r["loops"] == max_loops]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for bits in bits_set:
        s = sorted([r for r in sel if r["bits"] == bits], key=lambda r: r["loop_index"])
        xs = [r["loop_index"] for r in s]
        axes[0].plot(xs, [r["propagation_gain"] for r in s], marker="o", label=f"INT{bits}")
        axes[1].plot(xs, [r["cos_p_q"] for r in s], marker="o", label=f"INT{bits}")
    axes[0].set_title("propagation gain ||p||/||e||")
    axes[1].set_title("cos(p, q)")
    axes[1].axhline(0, color="k", lw=0.6)
    for ax in axes:
        ax.set_xlabel("loop index")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"TRM-LM error decomposition (loops={max_loops})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(v, d=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(v) or math.isinf(v):
        return "n/a"
    return f"{v:.{d}f}"


def write_report(path, rows, err_rows, meta):
    max_loops = max(r["loops"] for r in rows)
    lines = [
        "# Inference-time error compensation for the quantized TRM-LM",
        "",
        f"Frozen `TRM-LM-v2-10M`. Eval = WikiText-2 raw test, {meta['n_eval_predictions']} next-token predictions "
        f"(sha `{meta['eval_sha256'][:12]}`); calibration = the disjoint validation prefix. "
        "Weight-only asymmetric per-group RTN, group 128. Dequantized FP32 matmul, no low-bit kernel.",
        "",
        "**No compensator reads a full-precision hidden state at inference.** FP states build calibration "
        "targets and log prediction quality only. Mean and low-rank maps are fit sequentially on the "
        "compensated quantized rollout; dynamics is fit from the uncompensated rollout with one DAgger refit "
        "and free-runs at inference, seeded from the observable loop-0 state.",
        "",
        "## Error decomposition  e_{t+1} = p_t + q_t",
        "",
        f"At loops={max_loops}. p = propagation (previous error through the quantized core), "
        "q = injection (fresh quantization error on the FP state).",
        "",
        "| bits | loop | ‖e‖ | ‖p‖ | ‖q‖ | gain ‖p‖/‖e‖ | cos(p,q) | cos(e_t,e_{t+1}) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted([r for r in err_rows if r["loops"] == max_loops], key=lambda r: (r["bits"], r["loop_index"]), reverse=False):
        lines.append(
            f"| {r['bits']} | {r['loop_index']} | {_fmt(r['e_rms'])} | {_fmt(r['p_rms'])} | {_fmt(r['q_rms'])} | "
            f"{_fmt(r['propagation_gain'])} | {_fmt(r['cos_p_q'])} | {_fmt(r['cos_e_prev'])} |"
        )
    lines += ["", "## Compensators", "",
              "PPL at the largest loop count, with 95% bootstrap CI on CE. Fraction-of-oracle is how much of the "
              "naive→oracle CE gap each method closes.", "",
              "| bits | loops | method | PPL | CE [95% CI] | gap closed | frac oracle | pred cos | extra params |",
              "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (-r["bits"], r["loops"], r["method"])):
        if r["loops"] != max_loops:
            continue
        lines.append(
            f"| {r['bits']} | {r['loops']} | {r['method']} | {_fmt(r['ppl'], 2)} | "
            f"{_fmt(r['ce'], 4)} [{_fmt(r['ce_ci_lo'], 4)}, {_fmt(r['ce_ci_hi'], 4)}] | "
            f"{_fmt(100 * r['ce_gap_closed'], 1)}% | {_fmt(100 * r['fraction_of_oracle'], 1)}% | "
            f"{_fmt(r['pred_cosine'])} | {r['extra_params']} |"
        )
    lines += ["", "## Reading", "", _verdict(rows, err_rows), ""]
    path.write_text("\n".join(lines))
    print(f"wrote {path}", flush=True)


def _verdict(rows, err_rows):
    max_loops = max(r["loops"] for r in rows)
    best_bits = max(b for b in {r["bits"] for r in rows})
    msgs = []
    for bits in sorted({r["bits"] for r in rows}, reverse=True):
        sel = [r for r in rows if r["bits"] == bits and r["loops"] == max_loops]
        base = next((r for r in sel if r["method"] == "none"), None)
        if not base:
            continue
        learned = [r for r in sel if r["method"] not in ("none",)]
        if not learned:
            continue
        best = max(learned, key=lambda r: r["ce_gap_closed"] if not math.isnan(r["ce_gap_closed"]) else -1e9)
        msgs.append(
            f"INT{bits}: naive PPL {_fmt(base['ppl'],2)}, FP {_fmt(base['fp_ppl'],2)}, oracle {_fmt(base['oracle_ppl'],2)}; "
            f"best no-FP compensator = {best['method']} closing {_fmt(100*best['ce_gap_closed'],1)}% of the CE gap "
            f"({_fmt(100*best['fraction_of_oracle'],1)}% of oracle), pred cosine {_fmt(best['pred_cosine'])}."
        )
    return " ".join(msgs)


if __name__ == "__main__":
    run()
