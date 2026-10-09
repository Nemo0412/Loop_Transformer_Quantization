#!/usr/bin/env python3
"""Cross-loop quantization-error feedback on TRM-LM v2 10M.

Weight-only per-group RTN is simulated by dequantized FP32 weights. No integer
GEMM is used, so measured latency is not a low-bit kernel speedup.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from data import build_or_load_tokens
from fit import fit_low_rank, prediction_scores, predict_error
from forward import (
    forward_logits,
    forward_with_predictor,
    logits_from_state,
    loop_body,
    mse_sum,
    prelude,
    run_core,
    should_correct,
    token_ce_sum,
)
from modeling_trm2 import TRM2
from quantize import quantize_linear_weights_, quantize_weight, storage_summary

SEED = 42
LOOP_COUNTS = (1, 2, 3, 4, 6)
BITS = (4, 3, 2)
ALPHAS = (0.25, 0.5, 0.75, 1.0)
MODES = ("after_loop_1", "after_loop_2", "every_except_final")
RANKS = (4, 8, 16)
GROUP_SIZE = 128
SCHEME = "asymmetric"


class CEAcc:
    def __init__(self):
        self.nll = 0.0
        self.n = 0
        self.sq = 0.0
        self.m = 0

    def add(self, logits, ids, ref=None):
        nll, n = token_ce_sum(logits, ids)
        self.nll += nll
        self.n += n
        if ref is not None:
            sq, m = mse_sum(logits, ref)
            self.sq += sq
            self.m += m

    @property
    def ce(self) -> float:
        return self.nll / self.n

    @property
    def ppl(self) -> float:
        return math.exp(self.ce) if self.ce < 40 else float("inf")

    @property
    def mse(self) -> float:
        return self.sq / self.m if self.m else 0.0


class ResAcc:
    def __init__(self):
        self.final_sq = 0.0
        self.final_fp = 0.0
        self.final_n = 0
        self.all_sq = 0.0
        self.all_fp = 0.0
        self.all_n = 0

    def add(self, post, fp_states):
        final = (post[-1] - fp_states[-1]).float()
        self.final_sq += float(final.pow(2).sum().item())
        self.final_fp += float(fp_states[-1].float().pow(2).sum().item())
        self.final_n += final.numel()
        for h, ref in zip(post, fp_states):
            diff = (h - ref).float()
            self.all_sq += float(diff.pow(2).sum().item())
            self.all_fp += float(ref.float().pow(2).sum().item())
            self.all_n += diff.numel()

    @property
    def rms_final(self) -> float:
        return math.sqrt(self.final_sq / self.final_n)

    @property
    def rel_final(self) -> float:
        return self.rms_final / math.sqrt(self.final_fp / self.final_n)

    @property
    def rms_pooled(self) -> float:
        return math.sqrt(self.all_sq / self.all_n)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="two batches, INT4 only, write smoke_out/")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--calib-seqs", type=int, default=128)
    p.add_argument("--scheme", choices=("asymmetric", "symmetric"), default=SCHEME)
    p.add_argument("--exp4", choices=("auto", "yes", "no"), default="auto")
    p.add_argument("--prepare-only", action="store_true", help="build token windows and exit")
    return p.parse_args()


def load_model(path: Path, device: torch.device) -> TRM2:
    model = TRM2()
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    model.load_state_dict(obj)
    model.float().eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device)


def linear_names(model: nn.Module) -> set[str]:
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.add(name + ".weight")
    return names


def make_quantized(fp: TRM2, bits: int, scheme: str, device: torch.device):
    quant = copy.deepcopy(fp).cpu()
    infos = quantize_linear_weights_(quant, bits, GROUP_SIZE, scheme)
    if len(infos) != 28:
        raise RuntimeError(f"expected 28 Linear layers (prelude/core/coda), found {len(infos)}")
    regions = {i["region"] for i in infos}
    if regions != {"prelude", "core", "coda"}:
        raise RuntimeError(f"unexpected Linear regions: {regions}")
    names = linear_names(fp)
    for (n1, p1), (n2, p2) in zip(fp.named_parameters(), quant.named_parameters()):
        if n1 != n2:
            raise RuntimeError(f"parameter mismatch {n1} vs {n2}")
        if n1 not in names and not torch.equal(p1.cpu(), p2):
            raise RuntimeError(f"non-linear parameter changed: {n1}")
    for param in quant.parameters():
        param.requires_grad_(False)
    return quant.to(device).eval(), infos


def check_quant_math():
    weight = torch.randn(16, 704)
    errors = []
    for bits in (4, 3, 2):
        dequant, info = quantize_weight(weight, bits, GROUP_SIZE, "asymmetric")
        if dequant.shape != weight.shape or info["n_groups"] != 16 * 6:
            raise RuntimeError("group quantization shape/count check failed")
        errors.append(float((dequant - weight).norm()))
    if not (errors[0] < errors[1] < errors[2]):
        raise RuntimeError(f"quantization error did not increase as bits fell: {errors}")
    gen = torch.Generator().manual_seed(0)
    hidden = torch.randn(800, 32, generator=gen)
    u_true = torch.randn(32, 4, generator=gen)
    v_true = torch.randn(32, 4, generator=gen)
    target = predict_error(hidden, u_true, v_true)
    u_hat, v_hat = fit_low_rank(hidden, target, 4, ridge=1e-8)
    recovered = predict_error(hidden, u_hat, v_hat)
    rel = float((recovered - target).norm() / target.norm())
    if rel > 1e-2:
        raise RuntimeError(f"low-rank regression did not recover a rank-4 map (rel {rel})")


def check_forward(model, device):
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, model.c["VOCAB"], (2, 48), generator=gen, device="cpu").to(device)
    with torch.inference_mode():
        for loops in (1, 3, 6):
            ref = model(ids, loops=loops)
            got = forward_logits(model, ids, loops)
            err = float((ref - got).abs().max())
            if err > 1e-4:
                raise RuntimeError(f"loop-explicit forward differs at loops={loops}: max abs {err}")
            print(f"forward check loops={loops} max_abs={err:.3e}", flush=True)


def batches_of(tokens: torch.Tensor, batch_size: int, device, limit: int | None):
    n = tokens.shape[0] if limit is None else min(limit, tokens.shape[0])
    for start in range(0, n, batch_size):
        yield tokens[start : start + batch_size].to(device, non_blocking=True)


def time_call(fn, device, warmup: int, iters: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / iters
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) * 1000.0 / iters


def gap_closed(baseline: float, corrected: float, fp: float) -> float:
    gap = baseline - fp
    if abs(gap) < 1e-8:
        return float("nan")
    return (baseline - corrected) / gap


def correction_meta(mode: str, n_loops: int) -> dict | None:
    if mode == "after_loop_2" and n_loops < 2:
        return None
    if mode == "after_loop_1":
        subsequent, corrects_final = n_loops - 1, n_loops == 1
    elif mode == "after_loop_2":
        subsequent, corrects_final = n_loops - 2, n_loops == 2
    else:
        subsequent, corrects_final = (n_loops - 1 if n_loops > 1 else 0), False
    return {
        "n_subsequent_core_loops": subsequent,
        "corrects_final_core_state": corrects_final,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, float):
                    out[key] = "" if math.isnan(value) else f"{value:.8g}"
                elif isinstance(value, bool):
                    out[key] = "true" if value else "false"
                else:
                    out[key] = value
            writer.writerow(out)
    print(f"wrote {path}", flush=True)


def run_baseline(fp, quant_models, tokens, loops, batch_size, device, limit, time_batch, warmup, iters):
    stats = {loops_i: CEAcc() for loops_i in loops}
    qstats = {(bits, loops_i): CEAcc() for bits in quant_models for loops_i in loops}
    residual = {(bits, loops_i): ResAcc() for bits in quant_models for loops_i in loops}
    max_loops = max(loops)
    n_batches = math.ceil((tokens.shape[0] if limit is None else min(limit, tokens.shape[0])) / batch_size)

    with torch.inference_mode():
        for bi, ids in enumerate(batches_of(tokens, batch_size, device, limit)):
            if bi % 10 == 0:
                print(f"baseline batch {bi + 1}/{n_batches}", flush=True)
            e_fp, cos, sin = prelude(fp, ids)
            fp_states = run_core(fp, e_fp, e_fp, max_loops, cos, sin)
            fp_logits = {L: logits_from_state(fp, fp_states[L - 1], cos, sin) for L in loops}
            for L in loops:
                stats[L].add(fp_logits[L], ids)
            for bits, qmodel in quant_models.items():
                e_q, cos_q, sin_q = prelude(qmodel, ids)
                q_states = run_core(qmodel, e_q, e_q, max_loops, cos_q, sin_q)
                for L in loops:
                    q_logits = logits_from_state(qmodel, q_states[L - 1], cos_q, sin_q)
                    qstats[(bits, L)].add(q_logits, ids, fp_logits[L])
                    residual[(bits, L)].add(q_states[:L], fp_states[:L])
                    del q_logits
            del fp_logits

    latency = {}
    for L in loops:
        latency[("fp", L)] = time_call(lambda L=L: forward_logits(fp, time_batch, L), device, warmup, iters)
        for bits, qmodel in quant_models.items():
            latency[(bits, L)] = time_call(
                lambda qmodel=qmodel, L=L: forward_logits(qmodel, time_batch, L), device, warmup, iters
            )
    return stats, qstats, residual, latency


def run_error_analysis(fp, quant_models, tokens, n_loops, batch_size, device, limit):
    d = fp.c["D"]
    acc = {
        bits: [
            {
                "abs": 0.0,
                "sq": 0.0,
                "n": 0,
                "fp_sq": 0.0,
                "local_sq": 0.0,
                "prop_sq": 0.0,
                "cos_prev": 0.0,
                "cos_prev_n": 0,
                "cos_first": 0.0,
                "cos_first_n": 0,
                "cross_cos": 0.0,
                "cross_n": 0,
            }
            for _ in range(n_loops)
        ]
        for bits in quant_models
    }
    n_rows = tokens.shape[0] if limit is None else min(limit, tokens.shape[0])
    n_batches = math.ceil(n_rows / batch_size)
    checked = False

    with torch.inference_mode():
        for bi, ids in enumerate(batches_of(tokens, batch_size, device, limit)):
            if bi % 10 == 0:
                print(f"error-analysis batch {bi + 1}/{n_batches}", flush=True)
            e_fp, cos, sin = prelude(fp, ids)
            fp_states = run_core(fp, e_fp, e_fp, n_loops, cos, sin)
            for bits, qmodel in quant_models.items():
                e_q, _, _ = prelude(qmodel, ids)
                q_states = run_core(qmodel, e_q, e_q, n_loops, cos, sin)
                prev_err = None
                first_err = None
                for i in range(n_loops):
                    h_fp_prev = e_fp if i == 0 else fp_states[i - 1]
                    same_input = loop_body(qmodel, h_fp_prev, e_fp, i, cos, sin)
                    local = same_input - fp_states[i]
                    prop = q_states[i] - same_input
                    total = q_states[i] - fp_states[i]
                    if not checked:
                        if not torch.allclose(prop + local, total, atol=1e-4, rtol=1e-4):
                            raise RuntimeError("propagated + local error does not equal total error")
                    bucket = acc[bits][i]
                    diff = total.float()
                    bucket["abs"] += float(diff.abs().sum().item())
                    bucket["sq"] += float(diff.pow(2).sum().item())
                    bucket["n"] += diff.numel()
                    bucket["fp_sq"] += float(fp_states[i].float().pow(2).sum().item())
                    bucket["local_sq"] += float(local.float().pow(2).sum().item())
                    bucket["prop_sq"] += float(prop.float().pow(2).sum().item())
                    err = diff.reshape(-1, d)
                    if prev_err is not None:
                        bucket["cos_prev"] += _mean_cosine_sum(err, prev_err)
                        bucket["cos_prev_n"] += err.shape[0]
                    if first_err is not None:
                        bucket["cos_first"] += _mean_cosine_sum(err, first_err)
                        bucket["cos_first_n"] += err.shape[0]
                    bucket["cross_cos"] += _mean_cosine_sum(local.float().reshape(-1, d), prop.float().reshape(-1, d))
                    bucket["cross_n"] += err.shape[0]
                    prev_err = err
                    if first_err is None:
                        first_err = err
                checked = True
    rows = []
    for bits, series in acc.items():
        for i, bucket in enumerate(series):
            rms = math.sqrt(bucket["sq"] / bucket["n"])
            fp_rms = math.sqrt(bucket["fp_sq"] / bucket["n"])
            local_rms = math.sqrt(bucket["local_sq"] / bucket["n"])
            prop_rms = math.sqrt(bucket["prop_sq"] / bucket["n"])
            rows.append(
                {
                    "bits": bits,
                    "loop": i + 1,
                    "abs_mae": bucket["abs"] / bucket["n"],
                    "abs_rms": rms,
                    "rel_rms": rms / fp_rms,
                    "cos_with_prev": bucket["cos_prev"] / bucket["cos_prev_n"] if bucket["cos_prev_n"] else float("nan"),
                    "cos_with_first": bucket["cos_first"] / bucket["cos_first_n"] if bucket["cos_first_n"] else float("nan"),
                    "local_rms": local_rms,
                    "local_rel_rms": local_rms / fp_rms,
                    "propagated_rms": prop_rms,
                    "propagated_rel_rms": prop_rms / fp_rms,
                    "local_over_total_rms": local_rms / rms if rms > 0 else float("nan"),
                    "propagated_over_total_rms": prop_rms / rms if rms > 0 else float("nan"),
                    "local_propagated_cosine": bucket["cross_cos"] / bucket["cross_n"],
                    "n_tokens": bucket["n"] // d,
                    "trajectory_loops": n_loops,
                }
            )
    return rows


def _mean_cosine_sum(a: torch.Tensor, b: torch.Tensor) -> float:
    cos = (a * b).sum(-1) / (a.norm(dim=-1).clamp(min=1e-8) * b.norm(dim=-1).clamp(min=1e-8))
    return float(cos.sum().item())


def run_oracle(fp, quant_models, tokens, loops, batch_size, device, limit, baseline_lookup, residual_lookup, fp_ppl):
    max_loops = max(loops)
    keys = []
    for bits in quant_models:
        for mode in MODES:
            for alpha in ALPHAS:
                for L in loops:
                    if correction_meta(mode, L) is None:
                        continue
                    keys.append((bits, mode, alpha, L))
    acc = {key: CEAcc() for key in keys}
    resid = {key: ResAcc() for key in keys}
    n_rows = tokens.shape[0] if limit is None else min(limit, tokens.shape[0])
    n_batches = math.ceil(n_rows / batch_size)

    with torch.inference_mode():
        for bi, ids in enumerate(batches_of(tokens, batch_size, device, limit)):
            if bi % 10 == 0:
                print(f"oracle batch {bi + 1}/{n_batches}", flush=True)
            e_fp, cos_fp, sin_fp = prelude(fp, ids)
            fp_states = run_core(fp, e_fp, e_fp, max_loops, cos_fp, sin_fp)
            fp_logits = {L: logits_from_state(fp, fp_states[L - 1], cos_fp, sin_fp) for L in loops}
            for bits, qmodel in quant_models.items():
                for mode in ("after_loop_1", "after_loop_2"):
                    usable = [L for L in loops if correction_meta(mode, L) is not None]
                    if not usable:
                        continue
                    for alpha in ALPHAS:
                        post, cos_q, sin_q = _trajectory(qmodel, ids, max_loops, fp_states, mode, alpha)
                        for L in usable:
                            logits = logits_from_state(qmodel, post[L - 1], cos_q, sin_q)
                            key = (bits, mode, alpha, L)
                            acc[key].add(logits, ids, fp_logits[L])
                            resid[key].add(post[:L], fp_states[:L])
                            del logits
                for L in loops:
                    if L == 1:
                        continue
                    for alpha in ALPHAS:
                        post, cos_q, sin_q = _trajectory(qmodel, ids, L, fp_states[:L], "every_except_final", alpha)
                        logits = logits_from_state(qmodel, post[-1], cos_q, sin_q)
                        key = (bits, "every_except_final", alpha, L)
                        acc[key].add(logits, ids, fp_logits[L])
                        resid[key].add(post, fp_states[:L])
                        del logits
            del fp_logits

    rows = []
    for bits in quant_models:
        for mode in MODES:
            for L in loops:
                meta = correction_meta(mode, L)
                if meta is None:
                    continue
                base = baseline_lookup[(bits, L)]
                fp_ref = fp_ppl[L]
                zero_res = residual_lookup[(bits, L)]
                active = any(should_correct(mode, i, L) for i in range(L))
                if not active:
                    for alpha in (0.0,) + ALPHAS:
                        rows.append(
                            _oracle_row(
                                bits, mode, alpha, L, base["ce"], base["ppl"], base["logits_mse"],
                                base["ppl"], fp_ref, zero_res, meta,
                            )
                        )
                    continue
                rows.append(
                    _oracle_row(bits, mode, 0.0, L, base["ce"], base["ppl"], base["logits_mse"], base["ppl"], fp_ref, zero_res, meta)
                )
                for alpha in ALPHAS:
                    key = (bits, mode, alpha, L)
                    item = acc[key]
                    rows.append(
                        _oracle_row(bits, mode, alpha, L, item.ce, item.ppl, item.mse, base["ppl"], fp_ref, resid[key], meta)
                    )
    return rows


def _trajectory(model, ids, n_loops, fp_states, mode, alpha):
    e, cos, sin = prelude(model, ids)
    h = e
    post = []
    for i in range(n_loops):
        h = loop_body(model, h, e, i, cos, sin)
        if alpha != 0.0 and should_correct(mode, i, n_loops):
            h = h + alpha * (fp_states[i] - h)
        post.append(h)
    return post, cos, sin


def _oracle_row(bits, mode, alpha, loops, ce, ppl, mse, quant_ppl, fp_ppl, residual, meta):
    return {
        "bits": bits,
        "loops": loops,
        "mode": mode,
        "alpha": alpha,
        "ce": ce,
        "ppl": ppl,
        "logits_mse": mse,
        "ppl_gap_closed": gap_closed(quant_ppl, ppl, fp_ppl),
        "residual_rms_final": residual.rms_final,
        "residual_rel_final": residual.rel_final,
        "residual_rms_pooled": residual.rms_pooled,
        "n_subsequent_core_loops": meta["n_subsequent_core_loops"],
        "corrects_final_core_state": meta["corrects_final_core_state"],
        "fp_ppl": fp_ppl,
        "quant_ppl": quant_ppl,
    }


def oracle_is_meaningful(rows: list[dict]) -> tuple[bool, str]:
    best = -1.0
    best_desc = "none"
    for row in rows:
        if row["n_subsequent_core_loops"] < 1 or row["alpha"] == 0:
            continue
        gap = row["quant_ppl"] - row["fp_ppl"]
        rel_gap = gap / row["fp_ppl"] if row["fp_ppl"] else 0.0
        if gap < 0.05 and rel_gap < 0.01:
            continue
        recovery = row["ppl_gap_closed"]
        if math.isnan(recovery):
            continue
        if recovery > best:
            best = recovery
            best_desc = (
                f"INT{row['bits']} loops={row['loops']} mode={row['mode']} "
                f"alpha={row['alpha']} closes {recovery:.1%} of a PPL gap of {gap:.3f}"
            )
    if best >= 0.10:
        return True, f"Oracle recovery is meaningful: {best_desc}."
    if best < 0:
        return False, "No quantized setting had a perplexity gap large enough to score recovery."
    return False, f"Best cross-loop oracle recovery is only {best:.1%} ({best_desc}), below the 10% gate."


def collect_calibration(fp, qmodel, tokens, n_loops, batch_size, device):
    hidden = [[] for _ in range(n_loops)]
    error = [[] for _ in range(n_loops)]
    with torch.no_grad():
        for ids in batches_of(tokens, batch_size, device, None):
            e_fp, cos, sin = prelude(fp, ids)
            fp_states = run_core(fp, e_fp, e_fp, n_loops, cos, sin)
            e_q, cos_q, sin_q = prelude(qmodel, ids)
            q_states = run_core(qmodel, e_q, e_q, n_loops, cos_q, sin_q)
            for i in range(n_loops):
                hidden[i].append(q_states[i].reshape(-1, fp.c["D"]).float().cpu().clone())
                error[i].append((fp_states[i] - q_states[i]).reshape(-1, fp.c["D"]).float().cpu().clone())
    return [torch.cat(x, 0) for x in hidden], [torch.cat(x, 0) for x in error]


def run_predictor(
    fp,
    quant_models,
    eval_tokens,
    calib_tokens,
    loops,
    batch_size,
    device,
    limit,
    baseline_lookup,
    fp_ppl,
    latency,
    oracle_rows,
    warmup,
    iters,
    time_batch,
):
    max_loops = max(loops)
    rows = []
    saved = {}
    d = fp.c["D"]
    for bits, qmodel in quant_models.items():
        print(f"fitting INT{bits} correctors on calibration tokens", flush=True)
        hidden, error = collect_calibration(fp, qmodel, calib_tokens, max_loops, batch_size, device)
        shared = {}
        independent = {}
        h_cat = torch.cat(hidden, 0)
        e_cat = torch.cat(error, 0)
        for rank in RANKS:
            u, v = fit_low_rank(h_cat.to(device), e_cat.to(device), rank)
            shared[rank] = (u, v)
            independent[rank] = [
                fit_low_rank(hidden[i].to(device), error[i].to(device), rank) for i in range(max_loops)
            ]
            saved[f"int{bits}_shared_r{rank}_U"] = u.cpu()
            saved[f"int{bits}_shared_r{rank}_V"] = v.cpu()
            for i, (ui, vi) in enumerate(independent[rank]):
                saved[f"int{bits}_loop{i + 1}_r{rank}_U"] = ui.cpu()
                saved[f"int{bits}_loop{i + 1}_r{rank}_V"] = vi.cpu()
        del h_cat, e_cat

        for L in loops:
            base = baseline_lookup[(bits, L)]
            oracle = _oracle_ref(oracle_rows, bits, L)
            rows.append(
                _pred_row(
                    "none",
                    bits,
                    L,
                    "",
                    base,
                    base["ppl"],
                    fp_ppl[L],
                    oracle,
                    float("nan"),
                    float("nan"),
                    0,
                    0.0,
                    latency[(bits, L)],
                    latency[(bits, L)],
                    time_batch.shape[0] * time_batch.shape[1],
                )
            )
            if oracle is None:
                oracle_ppl = float("nan")
                oracle_mse = float("nan")
                oracle_ce = float("nan")
            else:
                oracle_ppl, oracle_mse, oracle_ce = oracle["ppl"], oracle["logits_mse"], oracle["ce"]
            rows.append(
                {
                    "status": "ok",
                    "method": "oracle",
                    "bits": bits,
                    "loops": L,
                    "rank": "",
                    "ce": oracle_ce,
                    "ppl": oracle_ppl,
                    "logits_mse": oracle_mse,
                    "ppl_gap_closed": gap_closed(base["ppl"], oracle_ppl, fp_ppl[L]) if oracle else float("nan"),
                    "oracle_ppl": oracle_ppl,
                    "fraction_of_oracle_recovery": 1.0 if oracle else float("nan"),
                    "pred_mse": float("nan"),
                    "pred_cosine": float("nan"),
                    "calib_pred_mse": float("nan"),
                    "calib_pred_cosine": float("nan"),
                    "n_extra_params": 0,
                    "extra_bytes": 0,
                    "correction_flops_per_token": "",
                    "latency_ms_per_token": "",
                    "uncorrected_latency_ms_per_token": latency[(bits, L)] / (time_batch.shape[0] * time_batch.shape[1]),
                    "latency_ratio": "",
                    "low_bit_kernel_used": False,
                    "reason": "diagnostic upper reference; uses FP hidden states and is not deployable",
                }
            )
            if L < 2:
                continue
            for rank in RANKS:
                for method, factors in (
                    ("shared", [shared[rank]] * L),
                    ("independent", independent[rank][:L]),
                ):
                    metrics = _eval_predictor(
                        fp, qmodel, eval_tokens, L, factors, batch_size, device, limit, fp_ppl_logits=True
                    )
                    used = L - 1
                    n_params = (1 if method == "shared" else used) * 2 * d * rank
                    ms = time_call(
                        lambda qmodel=qmodel, L=L, factors=factors: forward_with_predictor(
                            qmodel, time_batch, L, None, factors, False
                        ),
                        device,
                        warmup,
                        iters,
                    )
                    calib_mse, calib_cos = _calib_score(hidden, error, factors, method, L, device)
                    rows.append(
                        _pred_row(
                            method,
                            bits,
                            L,
                            rank,
                            metrics,
                            base["ppl"],
                            fp_ppl[L],
                            oracle,
                            calib_mse,
                            calib_cos,
                            n_params,
                            used * 4 * d * rank,
                            ms,
                            latency[(bits, L)],
                            time_batch.shape[0] * time_batch.shape[1],
                        )
                    )
        del hidden, error
    return rows, saved


def _oracle_ref(rows, bits, loops):
    matches = [
        r
        for r in rows
        if r["bits"] == bits and r["loops"] == loops and r["mode"] == "every_except_final" and r["alpha"] == 1.0
    ]
    return matches[0] if matches else None


def _eval_predictor(fp, qmodel, tokens, n_loops, factors, batch_size, device, limit, fp_ppl_logits):
    acc = CEAcc()
    pred_sq = pred_cos = pred_n = 0.0
    cos_n = 0
    d = fp.c["D"]
    with torch.inference_mode():
        for ids in batches_of(tokens, batch_size, device, limit):
            e_fp, cos, sin = prelude(fp, ids)
            fp_states = run_core(fp, e_fp, e_fp, n_loops, cos, sin)
            ref = logits_from_state(fp, fp_states[-1], cos, sin) if fp_ppl_logits else None
            logits, stats = forward_with_predictor(qmodel, ids, n_loops, fp_states, factors, True)
            acc.add(logits, ids, ref)
            pred_sq += stats["pred_sq"]
            pred_n += stats["pred_count"]
            # Recompute per-token cosine from the stored global sums is not possible.
            # forward_with_predictor stores a global dot; also accumulate cosine there.
            pred_cos += stats.get("cos_sum", 0.0)
            cos_n += stats.get("cos_n", 0)
            del logits, ref
    mse = pred_sq / pred_n if pred_n else float("nan")
    cosine = pred_cos / cos_n if cos_n else float("nan")
    return {"ce": acc.ce, "ppl": acc.ppl, "logits_mse": acc.mse, "pred_mse": mse, "pred_cosine": cosine}


def _calib_score(hidden, error, factors, method, n_loops, device):
    mses, cosines = [], []
    with torch.inference_mode():
        for i in range(n_loops - 1):
            u, v = factors[0] if method == "shared" else factors[i]
            # Score a stride subsample so the in-sample number stays cheap and stable.
            h = hidden[i][::8].to(device)
            e = error[i][::8].to(device)
            scores = prediction_scores(h, e, u, v)
            mses.append(scores["mse"])
            cosines.append(scores["cosine"])
    return sum(mses) / len(mses), sum(cosines) / len(cosines)


def _pred_row(method, bits, loops, rank, metrics, quant_ppl, fp_ppl_value, oracle, calib_mse, calib_cos, n_params, flops, ms, base_ms, tokens_per_batch):
    oracle_ppl = oracle["ppl"] if oracle else float("nan")
    method_ppl = metrics["ppl"] if isinstance(metrics, dict) else metrics
    method_ce = metrics["ce"] if isinstance(metrics, dict) else ""
    method_mse = metrics["logits_mse"] if isinstance(metrics, dict) else ""
    pred_mse = metrics.get("pred_mse", float("nan")) if isinstance(metrics, dict) else float("nan")
    pred_cos = metrics.get("pred_cosine", float("nan")) if isinstance(metrics, dict) else float("nan")
    oracle_gain = quant_ppl - oracle_ppl
    fraction = (quant_ppl - method_ppl) / oracle_gain if oracle and abs(oracle_gain) > 1e-8 else float("nan")
    ms_tok = ms / tokens_per_batch
    base_tok = base_ms / tokens_per_batch
    return {
        "status": "ok",
        "method": method,
        "bits": bits,
        "loops": loops,
        "rank": rank,
        "ce": method_ce,
        "ppl": method_ppl,
        "logits_mse": method_mse,
        "ppl_gap_closed": gap_closed(quant_ppl, method_ppl, fp_ppl_value),
        "oracle_ppl": oracle_ppl,
        "fraction_of_oracle_recovery": fraction,
        "pred_mse": pred_mse,
        "pred_cosine": pred_cos,
        "calib_pred_mse": calib_mse,
        "calib_pred_cosine": calib_cos,
        "n_extra_params": n_params,
        "extra_bytes": n_params * 4,
        "correction_flops_per_token": flops,
        "latency_ms_per_token": ms_tok,
        "uncorrected_latency_ms_per_token": base_tok,
        "latency_ratio": ms / base_ms if base_ms else float("nan"),
        "low_bit_kernel_used": False,
        "reason": "",
    }


def plot_errors(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    bits = sorted({int(r["bits"]) for r in rows}, reverse=True)
    colors = {4: "#1f77b4", 3: "#ff7f0e", 2: "#d62728"}
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True)
    panels = (
        (axes[0, 0], "abs_rms", "Total error RMS"),
        (axes[0, 1], "rel_rms", "Total error / FP RMS"),
        (axes[1, 0], "local_rms", "Local error RMS"),
        (axes[1, 1], "propagated_rms", "Propagated error RMS"),
    )
    for ax, key, title in panels:
        for b in bits:
            series = [r for r in rows if int(r["bits"]) == b]
            series.sort(key=lambda r: int(r["loop"]))
            ax.plot(
                [int(r["loop"]) for r in series],
                [float(r[key]) for r in series],
                marker="o",
                color=colors.get(b, "black"),
                label=f"INT{b}",
            )
        ax.set_title(title)
        ax.set_xlabel("Loop index")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("Quantization error at core-loop boundaries (fake per-group RTN)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def _fmt(x, digits=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.{digits}f}"


def write_report(path, ctx):
    base, errors, oracle, predictor = ctx["baseline"], ctx["errors"], ctx["oracle"], ctx["predictor"]
    fp = {(int(r["loops"])): r for r in base if r["method"] == "fp"}
    lines = []
    lines.append("# Cross-loop quantization error feedback")
    lines.append("")
    lines.append("Proof of concept on frozen `TRM-LM-v2-10M` (`trm2.pt`). The question is whether quantization error left by earlier passes through the shared core can be estimated and removed in later passes, without retraining the language model.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("The network is not a single shared matrix. `modeling_trm2.py` runs embedding, one prelude block, then `loops` passes of a five-block core, then one coda block. Each core pass is `h = RMS(h) * gain[i] + prelude + tag[i]` followed by the same five pre-norm attention/SwiGLU blocks. Those five blocks stay one module list: each Linear is quantized once and reused on every loop.")
    lines.append("")
    lines.append(
        f"Weight-only {ctx['scheme']} per-group round-to-nearest, group size {GROUP_SIZE}, along the input features of every `nn.Linear` in the prelude, core, and coda. The down projection has 704 input features, so its last group has 64 weights. Embeddings, the tied head, RMS scales, and the loop gain/tag stay FP32. Quantized weights are stored conceptually as packed integers plus FP16 scales"
        + (" and INT8 zero-points" if ctx["scheme"] == "asymmetric" else "")
        + ", but the forward pass multiplies the dequantized FP32 matrix. **No low-bit kernel is used.** TF32 is off. Latency below is that FP32 simulation, not an integer-GEMM speedup."
    )
    lines.append("")
    lines.append(
        f"Evaluation tokens are the WikiText-2 raw test split, packed in non-overlapping windows of {ctx['seq_len']} (sha256 `{ctx['meta']['eval_sha256'][:12]}`, {ctx['meta']['n_eval_predictions']} next-token predictions). Calibration for the linear corrector, when it is fit, is the first {ctx['meta']['n_calib_seq']} windows of the validation split. Seed {SEED}. Cross-entropy is in nats over the 16,384-token BPE; perplexity is `exp(CE)`."
    )
    lines.append("")
    lines.append(f"Device: {ctx['device_name']}.")
    lines.append("")
    lines.append("## Experiment 1: baseline RTN")
    lines.append("")
    lines.append("| method | loops | CE | PPL | logits MSE | storage MiB | ms/token |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in base:
        storage = row["storage_bytes"] / (1024**2) if row["method"] != "fp" else row["fp32_linear_bytes"] / (1024**2)
        lines.append(
            f"| {row['method']} | {row['loops']} | {_fmt(row['ce'], 4)} | {_fmt(row['ppl'], 3)} | {_fmt(row['logits_mse'], 4)} | {_fmt(storage, 3)} | {_fmt(row['latency_ms_per_token'], 4)} |"
        )
    lines.append("")
    lines.append("Storage is the theoretical packed size of the quantized Linear weights, including scales and zero-points, counted once even though the core is executed several times. The FP row uses the FP32 size of those same Linear weights.")
    fp_ms = [r["latency_ms_per_token"] for r in base if r["method"] == "fp"]
    q_ms = [r["latency_ms_per_token"] for r in base if r["method"] != "fp"]
    if fp_ms and q_ms:
        ratio = (sum(q_ms) / len(q_ms)) / (sum(fp_ms) / len(fp_ms))
        lines.append(
            f"Mean quantized-simulation latency is {ratio:.2f}× the FP32 latency. That ratio is kernel noise around a dense FP32 matmul, not a compression speedup."
        )
    lines.append("")
    lines.append("## Experiment 2: where the error comes from")
    lines.append("")
    lines.append("Hidden states are recorded after each five-block core pass. Local error is the one-step difference between a quantized core and an FP core on the same FP input state. Propagated error is the rest: the quantized core applied to the live quantized state, minus the quantized core applied to the FP state. The two parts add to the total error.")
    lines.append("")
    lines.append("| bits | loop | RMS | relative RMS | cosine vs previous | local/total | propagated/total |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in errors:
        lines.append(
            f"| {row['bits']} | {row['loop']} | {_fmt(row['abs_rms'], 4)} | {_fmt(row['rel_rms'], 4)} | {_fmt(row['cos_with_prev'], 3)} | {_fmt(row['local_over_total_rms'], 3)} | {_fmt(row['propagated_over_total_rms'], 3)} |"
        )
    lines.append("")
    lines.append("Plot: `plots/error_vs_loop.png`.")
    lines.append("")
    growth_bits = []
    for bits in sorted({int(r["bits"]) for r in errors}):
        series = sorted((r for r in errors if int(r["bits"]) == bits), key=lambda r: int(r["loop"]))
        if len(series) >= 2 and series[0]["abs_rms"] > 0:
            growth_bits.append(f"INT{bits} {series[-1]['abs_rms'] / series[0]['abs_rms']:.2f}×")
    if growth_bits:
        lines.append("Total RMS at the last loop relative to loop 1: " + ", ".join(growth_bits) + ".")
    cosines = [float(r["cos_with_prev"]) for r in errors if not math.isnan(float(r["cos_with_prev"]))]
    if cosines:
        lines.append(f"Mean consecutive-loop error cosine is {sum(cosines) / len(cosines):.3f}. A high value means the error direction is stable enough that a corrector could transfer across loops; a value near zero means each loop's error points somewhere else.")
    lines.append("")
    lines.append("## Experiment 3: oracle state correction")
    lines.append("")
    lines.append("This is an offline diagnostic. The FP hidden state is available only here. After a selected core boundary the run continues from `quantized + alpha * (fp - quantized)`, and every later core pass is quantized again. The injected error is recomputed from the live state, so an earlier correction changes the later target. Alpha 0 reproduces the uncorrected baseline.")
    lines.append("")
    lines.append("PPL at 6 loops, or at the largest loop count in this run:")
    lines.append("")
    focus_loops = 6 if any(int(r["loops"]) == 6 for r in oracle) else max(int(r["loops"]) for r in oracle)
    lines.append("| bits | mode | alpha | PPL | gap closed | final residual RMS | subsequent core loops |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    focus = [r for r in oracle if int(r["loops"]) == focus_loops and (float(r["alpha"]) in (0.0, 0.5, 1.0))]
    for row in focus:
        lines.append(
            f"| {row['bits']} | {row['mode']} | {_fmt(float(row['alpha']), 2)} | {_fmt(row['ppl'], 3)} | {_fmt(100 * row['ppl_gap_closed'], 1)}% | {_fmt(row['residual_rms_final'], 4)} | {row['n_subsequent_core_loops']} |"
        )
    lines.append("")
    lines.append(ctx["gate_reason"])
    lines.append("")
    lines.append("## Experiment 4: low-rank linear correction")
    lines.append("")
    if not ctx["ran_exp4"]:
        lines.append("Not run. The gate requires a cross-loop oracle (a correction with at least one later core pass) to close 10% of a real perplexity gap. " + ctx["gate_reason"])
        lines.append("")
        lines.append("`predictor_results.csv` records the skip. There is no claim that a learned corrector would help when the oracle does not.")
    else:
        lines.append("The corrector is `predicted_error = U @ V.T @ h`, fit by reduced-rank regression on validation pairs `(quantized state, fp state - quantized state)` only. Pretrained weights stay frozen. Test evaluation follows the corrected quantized trajectory. It does not teacher-force FP states. The oracle column is alpha-1 correction after every non-final loop.")
        lines.append("")
        lines.append("| bits | loops | method | rank | PPL | gap closed | fraction of oracle | pred MSE | pred cosine | extra params | latency ratio |")
        lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        shown = [
            r
            for r in predictor
            if r.get("status") == "ok" and int(r["loops"]) in (focus_loops, 3) and (r["rank"] in ("", 16, "16") or r["method"] in ("none", "oracle"))
        ]
        for row in shown:
            rank = row["rank"] if row["rank"] != "" else "-"
            lines.append(
                f"| {row['bits']} | {row['loops']} | {row['method']} | {rank} | {_fmt(float(row['ppl']), 3)} | {_fmt(100 * float(row['ppl_gap_closed']), 1)}% | {_fmt(float(row['fraction_of_oracle_recovery']), 2)} | {_fmt(float(row['pred_mse']) if row['pred_mse'] != '' else float('nan'), 5)} | {_fmt(float(row['pred_cosine']) if row['pred_cosine'] != '' else float('nan'), 3)} | {row['n_extra_params']} | {_fmt(float(row['latency_ratio']) if row['latency_ratio'] != '' else float('nan'), 3)} |"
            )
        lines.append("")
        lines.append("Latency ratio is corrected FP32 time over uncorrected quantized FP32 time. Extra FLOPs are `4 * D * rank` per corrected boundary per token (two rank-r matvecs). They are not measured inside an integer kernel.")
    lines.append("")
    lines.append("## Is cross-loop correction promising?")
    lines.append("")
    lines.append(ctx["verdict"])
    lines.append("")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}", flush=True)


def build_verdict(errors, oracle, predictor, ran_exp4) -> str:
    def series(bits):
        return sorted((r for r in errors if int(r["bits"]) == bits), key=lambda r: int(r["loop"]))

    bits_list = sorted({int(r["bits"]) for r in errors})
    growth = []
    local_frac = []
    for bits in bits_list:
        s = series(bits)
        if len(s) >= 2 and s[0]["abs_rms"] > 0:
            growth.append(s[-1]["abs_rms"] / s[0]["abs_rms"])
        if s:
            local_frac.append(s[-1]["local_over_total_rms"])
    cross = [
        r
        for r in oracle
        if r["mode"] == "every_except_final" and float(r["alpha"]) == 1.0 and int(r["n_subsequent_core_loops"]) >= 1
    ]
    best = max(cross, key=lambda r: r["ppl_gap_closed"] if not math.isnan(r["ppl_gap_closed"]) else -1e9, default=None)
    parts = []
    if growth:
        parts.append(
            f"Across bit-widths, total hidden-state RMS changes by {min(growth):.2f}× to {max(growth):.2f}× from the first loop to the last."
        )
    if local_frac:
        parts.append(
            f"At the last loop, local one-step error is {min(local_frac):.2f}–{max(local_frac):.2f}× the total RMS (values above 1 mean the propagated part points against the local part)."
        )
    if best is None or math.isnan(float(best["ppl_gap_closed"])):
        parts.append("The oracle did not produce a usable cross-loop comparison, so there is no evidence for a deployable corrector.")
        return " ".join(parts)
    parts.append(
        f"The strongest full-state cross-loop oracle is INT{best['bits']} at {best['loops']} loops, which closes {best['ppl_gap_closed']:.1%} of the uncorrected perplexity gap ({_fmt(best['quant_ppl'])} → {_fmt(best['ppl'])}, FP {_fmt(best['fp_ppl'])})."
    )
    if best["ppl_gap_closed"] < 0.10:
        parts.append(
            "That is not a meaningful recovery. Putting the hidden state back on the FP trajectory before later loops barely changes the final loss, so the perplexity gap is dominated by error that a cross-loop state correction never gets to remove. A learned feedback map is not justified by this evidence."
        )
        return " ".join(parts)
    if not ran_exp4:
        parts.append("The oracle has headroom, but the linear corrector was not fit in this run.")
        return " ".join(parts)
    learned = [r for r in predictor if r.get("method") in ("shared", "independent") and r.get("status") == "ok"]
    if not learned:
        parts.append("No learned-corrector rows were produced.")
        return " ".join(parts)
    top = max(learned, key=lambda r: r["ppl_gap_closed"] if not math.isnan(float(r["ppl_gap_closed"])) else -1e9)
    parts.append(
        f"The best rank-{top['rank']} {top['method']} corrector (INT{top['bits']}, {top['loops']} loops) closes {float(top['ppl_gap_closed']):.1%} of the FP gap and {float(top['fraction_of_oracle_recovery']):.0%} of what the oracle recovered, with {int(top['n_extra_params'])} extra FP32 parameters."
    )
    frac = float(top["fraction_of_oracle_recovery"])
    if frac >= 0.3 and float(top["ppl_gap_closed"]) >= 0.1:
        parts.append(
            "This is promising enough to study further: a frozen low-rank linear map, fit without touching the pretrained weights, captures a real share of the oracle gain. It remains a fake-quant result. It does not show that INT2/INT3/INT4 kernels would be faster, and it is not a training recipe for the 10M model."
        )
    elif frac >= 0.1:
        parts.append(
            "The oracle gap is partly real, and the linear map picks up a slice of it. That is a weak positive signal for cross-loop feedback, not enough to prefer it over simply keeping one more bit or one fewer loop. A nonlinear corrector would be the next measurement, still on frozen pretrained weights."
        )
    else:
        parts.append(
            "The oracle can move the loss, but this low-rank linear function of the hidden state does not predict the error well enough to keep that gain. Cross-loop correction is not promising in this linear form."
        )
    return " ".join(parts)


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    if not torch.cuda.is_available() and not (args.smoke or args.prepare_only):
        raise SystemExit("Full run needs a CUDA GPU. Re-run with --smoke for a CPU pipeline check.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    out = ROOT / "smoke_out" if args.smoke else ROOT
    loops = (1, 2) if args.smoke else LOOP_COUNTS
    bit_list = (4,) if args.smoke else BITS
    batch_limit = 2 if args.smoke else None
    warmup, iters = (1, 1) if args.smoke else (2, 8)
    print(f"device={device_name} scheme={args.scheme} smoke={args.smoke}", flush=True)

    check_quant_math()
    fp = load_model(ROOT / "checkpoints" / "trm2.pt", device)
    check_forward(fp, device)
    quant_models = {}
    storage = {}
    for bits in bit_list:
        qmodel, infos = make_quantized(fp, bits, args.scheme, device)
        quant_models[bits] = qmodel
        storage[bits] = storage_summary(infos)
        core_ptrs = {p.data_ptr() for name, p in qmodel.named_parameters() if name.startswith("core.") and name.endswith("weight")}
        print(
            f"INT{bits}: {storage[bits]['n_quant_params']} linear params, "
            f"storage {storage[bits]['storage_bytes']:.0f} bytes, "
            f"{len(core_ptrs)} distinct core weight tensors (shared across loops)",
            flush=True,
        )

    eval_tokens, calib_tokens, meta = build_or_load_tokens(
        ROOT, ROOT / "checkpoints" / "tokenizer.json", args.seq_len, args.calib_seqs, SEED
    )
    print(
        f"eval sequences={meta['n_eval_seq']} calib sequences={meta['n_calib_seq']} sha={meta['eval_sha256'][:12]}",
        flush=True,
    )
    if args.prepare_only:
        return
    time_batch = eval_tokens[: args.batch_size].to(device)

    fp_stats, q_stats, residual, latency = run_baseline(
        fp, quant_models, eval_tokens, loops, args.batch_size, device, batch_limit, time_batch, warmup, iters
    )
    baseline_rows = []
    tokens_per_batch = time_batch.shape[0] * time_batch.shape[1]
    for L in loops:
        baseline_rows.append(
            {
                "method": "fp",
                "bits": "",
                "loops": L,
                "scheme": "fp32",
                "group_size": "",
                "ce": fp_stats[L].ce,
                "ppl": fp_stats[L].ppl,
                "logits_mse": 0.0,
                "n_quant_params": 0,
                "packed_weight_bytes": 0,
                "scale_zp_bytes": 0,
                "storage_bytes": 0,
                "core_storage_bytes": 0,
                "fp32_linear_bytes": storage[bit_list[0]]["fp32_linear_bytes"],
                "latency_ms_per_batch": latency[("fp", L)],
                "latency_ms_per_token": latency[("fp", L)] / tokens_per_batch,
                "n_eval_tokens": fp_stats[L].n,
                "n_eval_predictions": fp_stats[L].n,
                "eval_sha256": meta["eval_sha256"],
                "quant_simulation": "fp32",
                "low_bit_kernel_used": False,
                "seed": SEED,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
            }
        )
    baseline_lookup = {}
    fp_ppl = {L: fp_stats[L].ppl for L in loops}
    for bits in bit_list:
        info = storage[bits]
        for L in loops:
            row = {
                "method": f"int{bits}",
                "bits": bits,
                "loops": L,
                "scheme": f"{args.scheme}_per_group_rtn",
                "group_size": GROUP_SIZE,
                "ce": q_stats[(bits, L)].ce,
                "ppl": q_stats[(bits, L)].ppl,
                "logits_mse": q_stats[(bits, L)].mse,
                "n_quant_params": info["n_quant_params"],
                "packed_weight_bytes": info["packed_weight_bytes"],
                "scale_zp_bytes": info["scale_zp_bytes"],
                "storage_bytes": info["storage_bytes"],
                "core_storage_bytes": info["core_storage_bytes"],
                "fp32_linear_bytes": info["fp32_linear_bytes"],
                "latency_ms_per_batch": latency[(bits, L)],
                "latency_ms_per_token": latency[(bits, L)] / tokens_per_batch,
                "n_eval_tokens": q_stats[(bits, L)].n,
                "n_eval_predictions": q_stats[(bits, L)].n,
                "eval_sha256": meta["eval_sha256"],
                "quant_simulation": info["quant_simulation"],
                "low_bit_kernel_used": False,
                "seed": SEED,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
            }
            baseline_rows.append(row)
            baseline_lookup[(bits, L)] = row
    if batch_limit is None and fp_stats[loops[0]].n != meta["n_eval_predictions"]:
        raise RuntimeError(
            f"prediction count {fp_stats[loops[0]].n} != prepared tokens {meta['n_eval_predictions']}"
        )
    base_fields = list(baseline_rows[0].keys())
    write_csv(out / "baseline_results.csv", baseline_rows, base_fields)

    error_rows = run_error_analysis(fp, quant_models, eval_tokens, max(loops), args.batch_size, device, batch_limit)
    error_fields = list(error_rows[0].keys())
    write_csv(out / "error_analysis.csv", error_rows, error_fields)
    plot_errors(error_rows, out / "plots" / "error_vs_loop.png")

    oracle_rows = run_oracle(
        fp, quant_models, eval_tokens, loops, args.batch_size, device, batch_limit, baseline_lookup, residual, fp_ppl
    )
    for row in oracle_rows:
        if float(row["alpha"]) == 0.0:
            base = baseline_lookup[(row["bits"], row["loops"])]
            if abs(row["ppl"] - base["ppl"]) > 1e-6:
                raise RuntimeError("alpha=0 oracle row does not match the baseline perplexity")
    oracle_fields = list(oracle_rows[0].keys())
    write_csv(out / "oracle_results.csv", oracle_rows, oracle_fields)

    meaningful, gate_reason = oracle_is_meaningful(oracle_rows)
    print(gate_reason, flush=True)
    ran_exp4 = args.exp4 == "yes" or (args.exp4 == "auto" and meaningful)
    pred_fields = [
        "status", "method", "bits", "loops", "rank", "ce", "ppl", "logits_mse", "ppl_gap_closed",
        "oracle_ppl", "fraction_of_oracle_recovery", "pred_mse", "pred_cosine", "calib_pred_mse",
        "calib_pred_cosine", "n_extra_params", "extra_bytes", "correction_flops_per_token",
        "latency_ms_per_token", "uncorrected_latency_ms_per_token", "latency_ratio",
        "low_bit_kernel_used", "reason",
    ]
    if ran_exp4:
        predictor_rows, saved = run_predictor(
            fp, quant_models, eval_tokens, calib_tokens, loops, args.batch_size, device, batch_limit,
            baseline_lookup, fp_ppl, latency, oracle_rows, warmup, iters, time_batch,
        )
        (out / "results").mkdir(parents=True, exist_ok=True)
        torch.save(saved, out / "results" / "correctors.pt")
    else:
        predictor_rows = [{
            "status": "skipped",
            "method": "",
            "bits": "",
            "loops": "",
            "rank": "",
            "ce": "",
            "ppl": "",
            "logits_mse": "",
            "ppl_gap_closed": "",
            "oracle_ppl": "",
            "fraction_of_oracle_recovery": "",
            "pred_mse": "",
            "pred_cosine": "",
            "calib_pred_mse": "",
            "calib_pred_cosine": "",
            "n_extra_params": "",
            "extra_bytes": "",
            "correction_flops_per_token": "",
            "latency_ms_per_token": "",
            "uncorrected_latency_ms_per_token": "",
            "latency_ratio": "",
            "low_bit_kernel_used": False,
            "reason": gate_reason,
        }]
    write_csv(out / "predictor_results.csv", predictor_rows, pred_fields)

    verdict = build_verdict(error_rows, oracle_rows, predictor_rows, ran_exp4)
    write_report(
        out / "report.md",
        {
            "baseline": baseline_rows,
            "errors": error_rows,
            "oracle": oracle_rows,
            "predictor": predictor_rows,
            "scheme": args.scheme,
            "meta": meta,
            "seq_len": args.seq_len,
            "device_name": device_name,
            "gate_reason": gate_reason,
            "ran_exp4": ran_exp4,
            "verdict": verdict,
        },
    )
    summary = {
        "device": device_name,
        "scheme": args.scheme,
        "low_bit_kernel_used": False,
        "quant_simulation": "fake_quant_dequantized_fp32_matmul",
        "ran_exp4": ran_exp4,
        "gate_reason": gate_reason,
        "eval_sha256": meta["eval_sha256"],
        "storage": {str(k): v for k, v in storage.items()},
    }
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(verdict, flush=True)


if __name__ == "__main__":
    main()
