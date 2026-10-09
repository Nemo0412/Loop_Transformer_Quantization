#!/usr/bin/env python3
"""Cross-loop quantization error feedback on the ARC Tiny Recursive Model.

Same protocol as the TRM-LM run, on arcprize/trm_arc_prize_verification.
The shared core is L_level (two blocks), reused inside every H cycle and every
ACT step. Weights stay frozen. Low-bit matmuls are not used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from arc_data import build_variant
from arc_model import (
    ARCTinyRecursiveModel,
    CastedLinear,
    load_arc_trm,
    one_h_cycle,
)
from fit import fit_low_rank, prediction_scores
from quantize import quantize_weight

SEED = 42
H_LIST = (1, 2, 3, 4, 6)
L_CYCLES = 4
ACT_STEPS = 16
BITS = (4, 3, 2)
ALPHAS = (0.25, 0.5, 0.75, 1.0)
MODES = ("after_loop_1", "after_loop_2", "every_except_final")
RANKS = (4, 8, 16)
GROUP = 128
VARIANTS = {
    "v1": "arc_v1_public/step_518071",
    "v2": "arc_v2_public/step_723914",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["v1", "v2"], choices=["v1", "v2"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-eval", type=int, default=96)
    parser.add_argument("--max-calib", type=int, default=64)
    parser.add_argument("--exp4", choices=("auto", "yes", "no"), default="auto")
    return parser.parse_args()


def log_stablemax(logits: torch.Tensor) -> torch.Tensor:
    shifted = torch.where(logits < 0, 1.0 / (1.0 - logits + 1e-30), logits + 1.0)
    return torch.log(shifted / shifted.sum(dim=-1, keepdim=True))


def token_stats(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    valid = labels != 0
    gathered = log_stablemax(logits.to(torch.float64)).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    nll = torch.where(valid, -gathered, 0.0)
    correct = valid & (logits.argmax(-1) == labels)
    valid_count = valid.sum(-1)
    exact = (valid_count > 0) & (correct.sum(-1) == valid_count)
    return {
        "nll": float(nll.sum().item()),
        "n_valid": int(valid.sum().item()),
        "n_correct": int(correct.sum().item()),
        "n_exact": int(exact.sum().item()),
        "n_seq": int((valid_count > 0).sum().item()),
    }


def add_stats(bucket: dict, stats: dict):
    for key, value in stats.items():
        bucket[key] = bucket.get(key, 0.0) + value


def finalize_stats(bucket: dict, mse_sq: float = 0.0, mse_n: int = 0) -> dict:
    ce = bucket["nll"] / bucket["n_valid"]
    return {
        "ce": ce,
        "ppl": math.exp(ce) if ce < 40 else float("inf"),
        "token_accuracy": bucket["n_correct"] / bucket["n_valid"],
        "exact_accuracy": bucket["n_exact"] / bucket["n_seq"],
        "logits_mse": mse_sq / mse_n if mse_n else 0.0,
        "n_valid": int(bucket["n_valid"]),
        "n_seq": int(bucket["n_seq"]),
    }


def mse_parts(pred, ref, labels) -> tuple[float, int]:
    valid = (labels != 0).unsqueeze(-1)
    diff = (pred.float() - ref.float()) * valid
    return float(diff.pow(2).sum().item()), int(valid.sum().item() * pred.shape[-1])


def should_correct(mode: str, index: int, n_h: int) -> bool:
    if mode == "after_loop_1":
        return index == 0
    if mode == "after_loop_2":
        return index == 1
    if mode == "every_except_final":
        return index < n_h - 1
    raise ValueError(mode)


def correction_meta(mode: str, n_h: int) -> dict | None:
    if mode == "after_loop_2" and n_h < 2:
        return None
    if mode == "after_loop_1":
        subsequent, corrects_final = n_h - 1, n_h == 1
    elif mode == "after_loop_2":
        subsequent, corrects_final = n_h - 2, n_h == 2
    else:
        subsequent, corrects_final = (n_h - 1 if n_h > 1 else 0), False
    return {"n_subsequent_h_cycles": subsequent, "corrects_final_h_state": corrects_final}


def prepare(model, inputs, puzzle_ids):
    embeddings = model.input_embeddings(inputs, puzzle_ids)
    rope = model.rotary_emb()
    z_h, z_l = model.init_state(inputs.shape[0], inputs.device)
    return embeddings, rope, z_h, z_l


def run_h(model, z_h, z_l, embeddings, rope, n_h: int):
    states_h, states_l = [], []
    for _ in range(n_h):
        z_h, z_l = one_h_cycle(model, z_h, z_l, embeddings, rope, L_CYCLES)
        states_h.append(z_h)
        states_l.append(z_l)
    return z_h, z_l, states_h, states_l


def forward_logits(model, inputs, puzzle_ids, n_h: int):
    embeddings, rope, z_h, z_l = prepare(model, inputs, puzzle_ids)
    for _ in range(ACT_STEPS):
        z_h, z_l, _, _ = run_h(model, z_h, z_l, embeddings, rope, n_h)
    return model.logits_from_z(z_h)


def quantize_copy(model: ARCTinyRecursiveModel, bits: int, device):
    clone = copy.deepcopy(model).cpu()
    infos = []
    for name, module in clone.named_modules():
        if not isinstance(module, CastedLinear):
            continue
        dequant, info = quantize_weight(module.weight.data, bits, GROUP, "asymmetric")
        module.weight.data.copy_(dequant)
        info["name"] = name
        info["region"] = "L_level" if name.startswith("L_level") else "head"
        infos.append(info)
    if len(infos) != 10:
        raise RuntimeError(f"expected 10 CastedLinear modules, found {len(infos)}")
    for param in clone.parameters():
        param.requires_grad_(False)
    return clone.to(device).eval(), infos


def storage_of(infos: list[dict]) -> dict:
    return {
        "n_quant_params": int(sum(i["n_params"] for i in infos)),
        "packed_weight_bytes": float(sum(i["packed_weight_bytes"] for i in infos)),
        "scale_zp_bytes": float(sum(i["scale_zp_bytes"] for i in infos)),
        "storage_bytes": float(sum(i["storage_bytes"] for i in infos)),
        "fp32_linear_bytes": float(sum(i["n_params"] for i in infos) * 4),
    }


def batches_of(data, batch_size, device, limit):
    n = min(limit, data["inputs"].shape[0])
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        yield (
            torch.tensor(data["inputs"][sl], device=device, dtype=torch.long),
            torch.tensor(data["labels"][sl], device=device, dtype=torch.long),
            torch.tensor(data["ids"][sl], device=device, dtype=torch.long),
        )


def time_call(fn, warmup, iters) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters


def gap_closed(base: float, new: float, fp: float) -> float:
    gap = base - fp
    if abs(gap) < 1e-8:
        return float("nan")
    return (base - new) / gap


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {key: "" for key in fields}
            for key, value in row.items():
                if isinstance(value, float):
                    out[key] = "" if math.isnan(value) else f"{value:.8g}"
                elif isinstance(value, bool):
                    out[key] = "true" if value else "false"
                else:
                    out[key] = value
            writer.writerow(out)
    print(f"wrote {path}", flush=True)


def slice_split(built, key, limit):
    return {
        "inputs": built[f"{key}_inputs"][:limit],
        "labels": built[f"{key}_labels"][:limit],
        "ids": built[f"{key}_ids"][:limit],
    }


def run_variant(variant: str, args, device, out_dir: Path):
    built = build_variant(ROOT, variant, ROOT / "data" / "arc_raw")
    ckpt = ensure_checkpoint(variant)
    fp = load_arc_trm(str(ckpt), torch.device("cpu"))
    n_ids = int(fp.puzzle_emb.weights.shape[0])
    if n_ids != built["meta"]["num_puzzle_identifiers"]:
        raise RuntimeError(
            f"{variant} puzzle-id count {built['meta']['num_puzzle_identifiers']} "
            f"does not match checkpoint rows {n_ids}"
        )
    eval_data = slice_split(built, "eval", args.max_eval)
    calib_data = slice_split(built, "calib", args.max_calib)
    digest = hashlib.sha256(np.ascontiguousarray(eval_data["inputs"]).tobytes()).hexdigest()
    print(
        f"{variant} eval {eval_data['inputs'].shape[0]} calib {calib_data['inputs'].shape[0]} "
        f"ids {n_ids} sha {digest[:12]}",
        flush=True,
    )
    quant_models = {}
    storage = {}
    for bits in BITS:
        quant_models[bits], infos = quantize_copy(fp, bits, device)
        storage[bits] = storage_of(infos)
        print(f"{variant} INT{bits} storage {storage[bits]['storage_bytes']:.0f} bytes", flush=True)
    ref_name = "L_level.layers.0.self_attn.qkv_proj.weight"
    ref = dict(fp.named_parameters())[ref_name].detach()
    errs = []
    for bits in BITS:
        qref = dict(quant_models[bits].named_parameters())[ref_name].detach().cpu()
        errs.append(float((qref - ref).pow(2).mean().item()))
    if not (errs[0] < errs[1] < errs[2]):
        raise RuntimeError(f"quant error did not increase as bits fell: {list(zip(BITS, errs))}")
    fp = fp.to(device)

    baseline_rows, baseline_lookup, fp_metrics, latency = run_baseline(
        variant, fp, quant_models, eval_data, storage, args.batch_size, device, digest
    )
    error_rows = run_errors(variant, fp, quant_models, eval_data, args.batch_size, device)
    plot_errors(error_rows, out_dir / "plots" / f"error_vs_loop_{variant}.png", variant)
    oracle_rows = run_oracle(
        variant, fp, quant_models, eval_data, args.batch_size, device, baseline_lookup, fp_metrics
    )
    meaningful, reason = oracle_is_meaningful(oracle_rows)
    print(variant, reason, flush=True)
    run_exp4 = args.exp4 == "yes" or (args.exp4 == "auto" and meaningful)
    if run_exp4:
        predictor_rows = run_predictor(
            variant, fp, quant_models, eval_data, calib_data, args.batch_size, device,
            baseline_lookup, fp_metrics, latency, oracle_rows,
        )
    else:
        predictor_rows = [{
            "checkpoint": variant, "status": "skipped", "method": "", "bits": "", "h_cycles": "",
            "rank": "", "reason": reason,
        }]
    del quant_models, fp
    torch.cuda.empty_cache()
    return baseline_rows, error_rows, oracle_rows, predictor_rows, meaningful, reason, run_exp4


def ensure_checkpoint(variant: str) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "arcprize/trm_arc_prize_verification",
        VARIANTS[variant],
        local_dir=str(ROOT / "checkpoints" / "arc"),
    )
    return Path(path)


def run_baseline(variant, fp, quant_models, data, storage, batch_size, device, digest):
    fp_buckets = {h: {} for h in H_LIST}
    q_buckets = {(bits, h): {} for bits in quant_models for h in H_LIST}
    mse = {(bits, h): [0.0, 0] for bits in quant_models for h in H_LIST}
    n_batches = math.ceil(data["inputs"].shape[0] / batch_size)
    with torch.inference_mode():
        for bi, (inputs, labels, puzzle_ids) in enumerate(batches_of(data, batch_size, device, data["inputs"].shape[0])):
            if bi % 5 == 0:
                print(f"{variant} baseline {bi + 1}/{n_batches}", flush=True)
            for h in H_LIST:
                fp_logits = forward_logits(fp, inputs, puzzle_ids, h)
                add_stats(fp_buckets[h], token_stats(fp_logits, labels))
                for bits, qmodel in quant_models.items():
                    q_logits = forward_logits(qmodel, inputs, puzzle_ids, h)
                    add_stats(q_buckets[(bits, h)], token_stats(q_logits, labels))
                    sq, n = mse_parts(q_logits, fp_logits, labels)
                    mse[(bits, h)][0] += sq
                    mse[(bits, h)][1] += n
                    del q_logits
                del fp_logits
    fp_metrics = {h: finalize_stats(fp_buckets[h]) for h in H_LIST}
    ids0, lab0, puz0 = next(batches_of(data, batch_size, device, batch_size))
    latency = {("fp", h): time_call(lambda h=h: forward_logits(fp, ids0, puz0, h), 1, 2) for h in H_LIST}
    for bits, qmodel in quant_models.items():
        for h in H_LIST:
            latency[(bits, h)] = time_call(lambda qmodel=qmodel, h=h: forward_logits(qmodel, ids0, puz0, h), 1, 2)
    rows = []
    lookup = {}
    per_example = ids0.shape[0]
    for h in H_LIST:
        rows.append(_base_row(variant, "fp", "", h, fp_metrics[h], storage[BITS[0]]["fp32_linear_bytes"], latency[("fp", h)], per_example, digest, "fp32"))
    for bits in quant_models:
        metrics = {h: finalize_stats(q_buckets[(bits, h)], mse[(bits, h)][0], mse[(bits, h)][1]) for h in H_LIST}
        for h in H_LIST:
            row = _base_row(
                variant, f"int{bits}", bits, h, metrics[h], storage[bits]["storage_bytes"],
                latency[(bits, h)], per_example, digest, "fake_quant_dequantized_fp32_matmul",
            )
            row.update(storage[bits])
            rows.append(row)
            lookup[(bits, h)] = row
    return rows, lookup, fp_metrics, latency


def _base_row(variant, method, bits, h, metrics, storage_bytes, latency_ms, n_example, digest, simulation):
    return {
        "checkpoint": variant,
        "method": method,
        "bits": bits,
        "h_cycles": h,
        "l_cycles": L_CYCLES,
        "act_steps": ACT_STEPS,
        "ce": metrics["ce"],
        "ppl": metrics["ppl"],
        "token_accuracy": metrics["token_accuracy"],
        "exact_accuracy": metrics["exact_accuracy"],
        "logits_mse": metrics["logits_mse"],
        "storage_bytes": storage_bytes,
        "latency_ms_per_example": latency_ms / n_example,
        "n_eval_seq": metrics["n_seq"],
        "n_eval_tokens": metrics["n_valid"],
        "eval_sha256": digest,
        "quant_simulation": simulation,
        "low_bit_kernel_used": False,
        "scheme": "asymmetric_per_group_rtn" if method != "fp" else "fp32",
        "group_size": GROUP if method != "fp" else "",
    }


def run_errors(variant, fp, quant_models, data, batch_size, device):
    n_h = max(H_LIST)
    acc = {
        bits: [dict(abs=0.0, sq=0.0, n=0, fp_sq=0.0, local_sq=0.0, prop_sq=0.0, zl_sq=0.0, cos_prev=0.0, cos_n=0, cross=0.0, cross_n=0) for _ in range(n_h)]
        for bits in quant_models
    }
    d = fp.config.hidden_size
    checked = False
    n_batches = math.ceil(data["inputs"].shape[0] / batch_size)
    with torch.inference_mode():
        for bi, (inputs, _labels, puzzle_ids) in enumerate(batches_of(data, batch_size, device, data["inputs"].shape[0])):
            if bi % 5 == 0:
                print(f"{variant} error {bi + 1}/{n_batches}", flush=True)
            emb_fp, rope_fp, z_h, z_l = prepare(fp, inputs, puzzle_ids)
            carry = {}
            for bits, qmodel in quant_models.items():
                carry[bits] = prepare(qmodel, inputs, puzzle_ids)
            for _step in range(ACT_STEPS):
                _, _, fp_hs, fp_ls = run_h(fp, z_h, z_l, emb_fp, rope_fp, n_h)
                rolled = {}
                for bits, qmodel in quant_models.items():
                    emb_q, rope_q, q_h, q_l = carry[bits]
                    _, _, q_hs, q_ls = run_h(qmodel, q_h, q_l, emb_q, rope_q, n_h)
                    rolled[bits] = (q_hs, q_ls, rope_q)
                h_prev, l_prev = z_h, z_l
                prev_err = {bits: None for bits in quant_models}
                for i in range(n_h):
                    for bits, qmodel in quant_models.items():
                        q_hs, q_ls, rope_q = rolled[bits]
                        local_h, _local_l = one_h_cycle(qmodel, h_prev, l_prev, emb_fp, rope_q, L_CYCLES)
                        total = q_hs[i] - fp_hs[i]
                        local = local_h - fp_hs[i]
                        prop = q_hs[i] - local_h
                        if not checked and not torch.allclose(prop + local, total, atol=1e-4, rtol=1e-4):
                            raise RuntimeError("error decomposition does not add up")
                        bucket = acc[bits][i]
                        diff = total.float()
                        bucket["abs"] += float(diff.abs().sum().item())
                        bucket["sq"] += float(diff.pow(2).sum().item())
                        bucket["n"] += diff.numel()
                        bucket["fp_sq"] += float(fp_hs[i].float().pow(2).sum().item())
                        bucket["local_sq"] += float(local.float().pow(2).sum().item())
                        bucket["prop_sq"] += float(prop.float().pow(2).sum().item())
                        bucket["zl_sq"] += float((q_ls[i] - fp_ls[i]).float().pow(2).sum().item())
                        err = diff.reshape(-1, d)
                        if prev_err[bits] is not None:
                            bucket["cos_prev"] += _cosine_sum(err, prev_err[bits])
                            bucket["cos_n"] += err.shape[0]
                        bucket["cross"] += _cosine_sum(local.float().reshape(-1, d), prop.float().reshape(-1, d))
                        bucket["cross_n"] += err.shape[0]
                        prev_err[bits] = err
                    h_prev, l_prev = fp_hs[i], fp_ls[i]
                    checked = True
                z_h, z_l = fp_hs[-1], fp_ls[-1]
                for bits in quant_models:
                    emb_q, _rope, _qh, _ql = carry[bits]
                    q_hs, q_ls, rope_q = rolled[bits]
                    carry[bits] = (emb_q, rope_q, q_hs[-1], q_ls[-1])
    rows = []
    for bits, series in acc.items():
        for i, bucket in enumerate(series):
            rms = math.sqrt(bucket["sq"] / bucket["n"])
            fp_rms = math.sqrt(bucket["fp_sq"] / bucket["n"])
            local_rms = math.sqrt(bucket["local_sq"] / bucket["n"])
            prop_rms = math.sqrt(bucket["prop_sq"] / bucket["n"])
            rows.append({
                "checkpoint": variant,
                "bits": bits,
                "h_cycle": i + 1,
                "abs_mae": bucket["abs"] / bucket["n"],
                "abs_rms": rms,
                "rel_rms": rms / fp_rms,
                "z_l_rms": math.sqrt(bucket["zl_sq"] / bucket["n"]),
                "cos_with_prev": bucket["cos_prev"] / bucket["cos_n"] if bucket["cos_n"] else float("nan"),
                "local_rms": local_rms,
                "propagated_rms": prop_rms,
                "local_over_total_rms": local_rms / rms if rms else float("nan"),
                "propagated_over_total_rms": prop_rms / rms if rms else float("nan"),
                "local_propagated_cosine": bucket["cross"] / bucket["cross_n"],
                "trajectory_h_cycles": n_h,
                "act_steps": ACT_STEPS,
            })
    return rows


def _cosine_sum(a, b) -> float:
    cos = (a * b).sum(-1) / (a.norm(dim=-1).clamp(min=1e-8) * b.norm(dim=-1).clamp(min=1e-8))
    return float(cos.sum().item())


def _active_modes(n_h: int) -> list[str]:
    return [mode for mode in MODES if correction_meta(mode, n_h) is not None and any(should_correct(mode, i, n_h) for i in range(n_h))]


def fp_rollout(model, inputs, puzzle_ids, n_h: int):
    embeddings, rope, z_h, z_l = prepare(model, inputs, puzzle_ids)
    steps = []
    for _step in range(ACT_STEPS):
        z_h, z_l, states_h, states_l = run_h(model, z_h, z_l, embeddings, rope, n_h)
        steps.append((states_h, states_l))
    return steps, model.logits_from_z(z_h)


def quant_rollout(model, inputs, puzzle_ids, n_h: int, steps, mode: str, alpha: float, factors=None):
    embeddings, rope, z_h, z_l = prepare(model, inputs, puzzle_ids)
    pred_sq = pred_cos = 0.0
    pred_n = cos_n = 0
    d = z_h.shape[-1]
    for step, (fp_hs, fp_ls) in enumerate(steps):
        for i in range(n_h):
            z_h, z_l = one_h_cycle(model, z_h, z_l, embeddings, rope, L_CYCLES)
            if factors is not None and i < n_h - 1:
                (uh, vh), (ul, vl) = factors[i]
                pred_h = (z_h @ vh) @ uh.T
                target_h = fp_hs[i] - z_h
                diff = (pred_h - target_h).float()
                pred_sq += float(diff.pow(2).sum().item())
                pred_n += diff.numel()
                pred_cos += _cosine_sum(pred_h.float().reshape(-1, d), target_h.float().reshape(-1, d))
                cos_n += pred_h.shape[0] * pred_h.shape[1]
                z_h = z_h + pred_h
                z_l = z_l + (z_l @ vl) @ ul.T
            elif alpha != 0.0 and should_correct(mode, i, n_h):
                z_h = z_h + alpha * (fp_hs[i] - z_h)
                z_l = z_l + alpha * (fp_ls[i] - z_l)
        if step == ACT_STEPS - 1:
            final_fp = fp_hs[-1]
            diff = (z_h - final_fp).float()
            return (
                model.logits_from_z(z_h),
                float(diff.pow(2).sum().item()),
                diff.numel(),
                pred_sq,
                pred_cos,
                pred_n,
                cos_n,
            )
    raise RuntimeError("no ACT steps")


def run_oracle(variant, fp, quant_models, data, batch_size, device, lookup, fp_metrics):
    keys = [
        (bits, mode, alpha, h)
        for bits in quant_models
        for h in H_LIST
        for mode in _active_modes(h)
        for alpha in ALPHAS
    ]
    buckets = {key: {} for key in keys}
    mse = {key: [0.0, 0] for key in keys}
    residual = {key: [0.0, 0] for key in keys}
    n_batches = math.ceil(data["inputs"].shape[0] / batch_size)
    with torch.inference_mode():
        probe_in, _probe_lab, probe_ids = next(batches_of(data, batch_size, device, batch_size))
        plain = forward_logits(quant_models[BITS[0]], probe_in, probe_ids, 3)
        steps, _fp_logits = fp_rollout(fp, probe_in, probe_ids, 3)
        echoed, *_rest = quant_rollout(quant_models[BITS[0]], probe_in, probe_ids, 3, steps, "every_except_final", 0.0)
        if not torch.allclose(plain, echoed, atol=1e-5, rtol=1e-5):
            raise RuntimeError("alpha 0 oracle diverges from the uncorrected quantized forward")
        del plain, echoed, steps
        for bi, (inputs, labels, puzzle_ids) in enumerate(batches_of(data, batch_size, device, data["inputs"].shape[0])):
            if bi % 5 == 0:
                print(f"{variant} oracle {bi + 1}/{n_batches}", flush=True)
            for h in H_LIST:
                modes = _active_modes(h)
                if not modes:
                    continue
                steps, fp_logits = fp_rollout(fp, inputs, puzzle_ids, h)
                for bits, qmodel in quant_models.items():
                    for mode in modes:
                        for alpha in ALPHAS:
                            q_logits, res_sq, res_n, *_pred = quant_rollout(
                                qmodel, inputs, puzzle_ids, h, steps, mode, alpha
                            )
                            key = (bits, mode, alpha, h)
                            add_stats(buckets[key], token_stats(q_logits, labels))
                            sq, n = mse_parts(q_logits, fp_logits, labels)
                            mse[key][0] += sq
                            mse[key][1] += n
                            residual[key][0] += res_sq
                            residual[key][1] += res_n
                            del q_logits
                del steps, fp_logits
    rows = []
    for bits in quant_models:
        for mode in MODES:
            for h in H_LIST:
                meta = correction_meta(mode, h)
                if meta is None:
                    continue
                base = lookup[(bits, h)]
                fp_ref = fp_metrics[h]
                active = any(should_correct(mode, i, h) for i in range(h))
                alphas = (0.0,) if not active else (0.0,) + ALPHAS
                for alpha in alphas:
                    if alpha == 0.0 or not active:
                        metrics = {
                            "ce": base["ce"], "ppl": base["ppl"], "token_accuracy": base["token_accuracy"],
                            "exact_accuracy": base["exact_accuracy"], "logits_mse": base["logits_mse"],
                        }
                        res_rms = float("nan")
                    else:
                        key = (bits, mode, alpha, h)
                        metrics = finalize_stats(buckets[key], mse[key][0], mse[key][1])
                        res_rms = math.sqrt(residual[key][0] / residual[key][1])
                    rows.append({
                        "checkpoint": variant,
                        "bits": bits,
                        "h_cycles": h,
                        "mode": mode,
                        "alpha": alpha,
                        "ce": metrics["ce"],
                        "ppl": metrics["ppl"],
                        "token_accuracy": metrics.get("token_accuracy", base["token_accuracy"]),
                        "exact_accuracy": metrics.get("exact_accuracy", base["exact_accuracy"]),
                        "logits_mse": metrics["logits_mse"],
                        "ce_gap_closed": gap_closed(base["ce"], metrics["ce"], fp_ref["ce"]),
                        "exact_accuracy_gain": metrics.get("exact_accuracy", base["exact_accuracy"]) - base["exact_accuracy"],
                        "residual_rms_final_z_h": res_rms,
                        "fp_ce": fp_ref["ce"],
                        "quant_ce": base["ce"],
                        "fp_exact_accuracy": fp_ref["exact_accuracy"],
                        "quant_exact_accuracy": base["exact_accuracy"],
                        **meta,
                        "act_steps": ACT_STEPS,
                    })
    return rows


def _gap_is_real(row: dict) -> bool:
    gap = float(row["quant_ce"]) - float(row["fp_ce"])
    base = float(row["fp_ce"])
    return gap >= 0.05 or (base > 1e-8 and gap / base >= 0.01)


def oracle_is_meaningful(rows: list[dict]) -> tuple[bool, str]:
    best = -1.0
    best_desc = "none"
    best_acc = -1.0
    acc_desc = "none"
    scored = False
    for row in rows:
        if row["alpha"] == 0 or row["n_subsequent_h_cycles"] < 1:
            continue
        recovery = row["ce_gap_closed"]
        acc_gain = float(row["exact_accuracy_gain"])
        acc_room = float(row["fp_exact_accuracy"]) - float(row["quant_exact_accuracy"])
        if acc_room >= 0.01 and acc_gain > best_acc:
            best_acc = acc_gain
            acc_desc = (
                f"INT{row['bits']} H={row['h_cycles']} {row['mode']} alpha={row['alpha']} "
                f"raises exact accuracy by {acc_gain:+.3f}"
            )
        if math.isnan(recovery) or not _gap_is_real(row):
            continue
        scored = True
        if recovery > best:
            best = recovery
            best_desc = (
                f"INT{row['bits']} H={row['h_cycles']} {row['mode']} alpha={row['alpha']} "
                f"closes {recovery:.1%} of the CE gap and changes exact accuracy by {acc_gain:+.3f}"
            )
    if best >= 0.10:
        return True, f"Oracle recovery is meaningful: {best_desc}."
    if best_acc >= 0.02:
        return True, f"Oracle recovery is meaningful: {acc_desc}."
    if not scored and best_acc < 0:
        return False, "No H-cycle setting had a cross-entropy gap large enough to score."
    return False, f"Best cross-loop oracle CE recovery is {best:.1%} ({best_desc}), below the 10% gate."


def _fit_factors(hidden_h, error_h, hidden_l, error_l, device):
    shared, independent = {}, {}
    for rank in RANKS:
        shared[rank] = (
            fit_low_rank(torch.cat(hidden_h, 0).to(device), torch.cat(error_h, 0).to(device), rank),
            fit_low_rank(torch.cat(hidden_l, 0).to(device), torch.cat(error_l, 0).to(device), rank),
        )
        independent[rank] = [
            (
                fit_low_rank(hidden_h[i].to(device), error_h[i].to(device), rank),
                fit_low_rank(hidden_l[i].to(device), error_l[i].to(device), rank),
            )
            for i in range(max(H_LIST) - 1)
        ]
    return shared, independent


def run_predictor(variant, fp, quant_models, eval_data, calib_data, batch_size, device, lookup, fp_metrics, latency, oracle_rows):
    d = fp.config.hidden_size
    fitted = {}
    for bits, qmodel in quant_models.items():
        print(f"{variant} fitting INT{bits}", flush=True)
        hidden_h, error_h, hidden_l, error_l = collect_pairs(fp, qmodel, calib_data, batch_size, device)
        fitted[bits] = _fit_factors(hidden_h, error_h, hidden_l, error_l, device)
        del hidden_h, error_h, hidden_l, error_l
    keys = [
        (bits, h, method, rank)
        for bits in quant_models
        for h in H_LIST if h >= 2
        for rank in RANKS
        for method in ("shared", "independent")
    ]
    buckets = {key: {} for key in keys}
    mse = {key: [0.0, 0] for key in keys}
    pred = {key: [0.0, 0.0, 0, 0] for key in keys}
    n_batches = math.ceil(eval_data["inputs"].shape[0] / batch_size)
    with torch.inference_mode():
        for bi, (inputs, labels, puzzle_ids) in enumerate(batches_of(eval_data, batch_size, device, eval_data["inputs"].shape[0])):
            if bi % 5 == 0:
                print(f"{variant} predictor {bi + 1}/{n_batches}", flush=True)
            for h in H_LIST:
                if h < 2:
                    continue
                steps, fp_logits = fp_rollout(fp, inputs, puzzle_ids, h)
                for bits, qmodel in quant_models.items():
                    shared, independent = fitted[bits]
                    for rank in RANKS:
                        for method in ("shared", "independent"):
                            factors = _factors_for(shared, independent, method, rank, h)
                            q_logits, _res_sq, _res_n, pred_sq, pred_cos, pred_n, cos_n = quant_rollout(
                                qmodel, inputs, puzzle_ids, h, steps, "every_except_final", 0.0, factors
                            )
                            key = (bits, h, method, rank)
                            add_stats(buckets[key], token_stats(q_logits, labels))
                            sq, n = mse_parts(q_logits, fp_logits, labels)
                            mse[key][0] += sq
                            mse[key][1] += n
                            pred[key][0] += pred_sq
                            pred[key][1] += pred_cos
                            pred[key][2] += pred_n
                            pred[key][3] += cos_n
                            del q_logits
                del steps, fp_logits
    per_example = batch_size
    ids0, _lab0, puz0 = next(batches_of(eval_data, batch_size, device, batch_size))
    rows = []
    for bits, qmodel in quant_models.items():
        shared, independent = fitted[bits]
        for h in H_LIST:
            base = lookup[(bits, h)]
            oracle = _oracle_ref(oracle_rows, variant, bits, h)
            rows.append(_pred_row(
                variant, "none", bits, h, "", base, base, fp_metrics[h], oracle,
                float("nan"), float("nan"), 0, 0.0, latency[(bits, h)], latency[(bits, h)], per_example,
            ))
            rows.append(_oracle_pred_row(variant, bits, h, base, fp_metrics[h], oracle, latency[(bits, h)], per_example))
            if h < 2:
                continue
            for rank in RANKS:
                for method in ("shared", "independent"):
                    key = (bits, h, method, rank)
                    metrics = finalize_stats(buckets[key], mse[key][0], mse[key][1])
                    factors = _factors_for(shared, independent, method, rank, h)
                    n_maps = (1 if method == "shared" else h - 1) * 2
                    n_params = n_maps * 2 * d * rank
                    flops = ACT_STEPS * (h - 1) * 8 * d * rank
                    ms = time_call(
                        lambda qmodel=qmodel, h=h, factors=factors: apply_predictor_logits(qmodel, ids0, puz0, h, factors),
                        1, 2,
                    )
                    rows.append(_pred_row(
                        variant, method, bits, h, rank, metrics, base, fp_metrics[h], oracle,
                        pred[key][0] / pred[key][2], pred[key][1] / pred[key][3],
                        n_params, flops, ms, latency[(bits, h)], per_example,
                    ))
    return rows


def _factors_for(shared, independent, method, rank, n_h):
    if method == "shared":
        return [shared[rank]] * (n_h - 1)
    return independent[rank][: n_h - 1]


def collect_pairs(fp, qmodel, data, batch_size, device):
    n_h = max(H_LIST)
    hidden_h = [[] for _ in range(n_h - 1)]
    error_h = [[] for _ in range(n_h - 1)]
    hidden_l = [[] for _ in range(n_h - 1)]
    error_l = [[] for _ in range(n_h - 1)]
    with torch.inference_mode():
        for inputs, _labels, puzzle_ids in batches_of(data, batch_size, device, data["inputs"].shape[0]):
            emb_fp, rope_fp, fp_h, fp_l = prepare(fp, inputs, puzzle_ids)
            emb_q, rope_q, q_h, q_l = prepare(qmodel, inputs, puzzle_ids)
            for _step in range(ACT_STEPS):
                _, _, fp_hs, fp_ls = run_h(fp, fp_h, fp_l, emb_fp, rope_fp, n_h)
                _, _, q_hs, q_ls = run_h(qmodel, q_h, q_l, emb_q, rope_q, n_h)
                for i in range(n_h - 1):
                    hidden_h[i].append(q_hs[i][:, ::32].reshape(-1, fp.config.hidden_size).float().cpu())
                    error_h[i].append((fp_hs[i] - q_hs[i])[:, ::32].reshape(-1, fp.config.hidden_size).float().cpu())
                    hidden_l[i].append(q_ls[i][:, ::32].reshape(-1, fp.config.hidden_size).float().cpu())
                    error_l[i].append((fp_ls[i] - q_ls[i])[:, ::32].reshape(-1, fp.config.hidden_size).float().cpu())
                fp_h, fp_l = fp_hs[-1], fp_ls[-1]
                q_h, q_l = q_hs[-1], q_ls[-1]
    return [torch.cat(x, 0) for x in hidden_h], [torch.cat(x, 0) for x in error_h], [torch.cat(x, 0) for x in hidden_l], [torch.cat(x, 0) for x in error_l]


def apply_predictor_logits(model, inputs, puzzle_ids, n_h, factors):
    embeddings, rope, z_h, z_l = prepare(model, inputs, puzzle_ids)
    for _step in range(ACT_STEPS):
        for i in range(n_h):
            z_h, z_l = one_h_cycle(model, z_h, z_l, embeddings, rope, L_CYCLES)
            if i < n_h - 1:
                (uh, vh), (ul, vl) = factors[i]
                z_h = z_h + (z_h @ vh) @ uh.T
                z_l = z_l + (z_l @ vl) @ ul.T
    return model.logits_from_z(z_h)


def _oracle_ref(rows, variant, bits, h):
    matches = [
        r for r in rows
        if r["checkpoint"] == variant and r["bits"] == bits and r["h_cycles"] == h
        and r["mode"] == "every_except_final" and float(r["alpha"]) == 1.0
    ]
    return matches[0] if matches else None


def _pred_row(variant, method, bits, h, rank, metrics, base, fp_metrics, oracle, pred_mse, pred_cos, n_params, flops, ms, base_ms, n_example):
    oracle_ce = oracle["ce"] if oracle else float("nan")
    return {
        "checkpoint": variant,
        "status": "ok",
        "method": method,
        "bits": bits,
        "h_cycles": h,
        "rank": rank,
        "ce": metrics["ce"],
        "ppl": metrics["ppl"],
        "exact_accuracy": metrics["exact_accuracy"],
        "logits_mse": metrics["logits_mse"],
        "ce_gap_closed": gap_closed(base["ce"], metrics["ce"], fp_metrics["ce"]),
        "fraction_of_oracle_recovery": (base["ce"] - metrics["ce"]) / (base["ce"] - oracle_ce) if oracle and abs(base["ce"] - oracle_ce) > 1e-8 else float("nan"),
        "pred_mse": pred_mse,
        "pred_cosine": pred_cos,
        "n_extra_params": n_params,
        "extra_bytes": n_params * 4,
        "correction_flops_per_token": flops,
        "latency_ms_per_example": ms / n_example,
        "latency_ratio": ms / base_ms if base_ms else float("nan"),
        "low_bit_kernel_used": False,
        "reason": "",
    }


def _oracle_pred_row(variant, bits, h, base, fp_metrics, oracle, base_ms, n_example):
    return {
        "checkpoint": variant,
        "status": "ok",
        "method": "oracle",
        "bits": bits,
        "h_cycles": h,
        "rank": "",
        "ce": oracle["ce"] if oracle else float("nan"),
        "ppl": oracle["ppl"] if oracle else float("nan"),
        "exact_accuracy": oracle["exact_accuracy"] if oracle else float("nan"),
        "logits_mse": oracle["logits_mse"] if oracle else float("nan"),
        "ce_gap_closed": oracle["ce_gap_closed"] if oracle else float("nan"),
        "fraction_of_oracle_recovery": 1.0 if oracle else float("nan"),
        "pred_mse": float("nan"),
        "pred_cosine": float("nan"),
        "n_extra_params": 0,
        "extra_bytes": 0,
        "correction_flops_per_token": "",
        "latency_ms_per_example": "",
        "latency_ratio": "",
        "low_bit_kernel_used": False,
        "reason": "diagnostic; needs the FP hidden state and is not deployable",
    }


def plot_errors(rows, path: Path, variant: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {4: "#1f77b4", 3: "#ff7f0e", 2: "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
    for bits in (4, 3, 2):
        series = sorted((r for r in rows if int(r["bits"]) == bits), key=lambda r: int(r["h_cycle"]))
        if not series:
            continue
        xs = [int(r["h_cycle"]) for r in series]
        axes[0].plot(xs, [float(r["abs_rms"]) for r in series], marker="o", color=colors[bits], label=f"INT{bits}")
        axes[1].plot(xs, [float(r["rel_rms"]) for r in series], marker="o", color=colors[bits], label=f"INT{bits}")
    axes[0].set_title("z_H error RMS")
    axes[1].set_title("z_H error / FP RMS")
    for ax in axes:
        ax.set_xlabel("H-cycle index")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"ARC TRM {variant} quantization error across H cycles")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(value, digits=3):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(value) or math.isinf(value):
        return "n/a"
    return f"{value:.{digits}f}"


def write_report(path: Path, ctx: dict):
    lines = [
        "# Cross-loop quantization on the ARC Tiny Recursive Model",
        "",
        "Same question as the TRM-LM run, now on the frozen ARC checkpoints from [arcprize/trm_arc_prize_verification](https://huggingface.co/arcprize/trm_arc_prize_verification). No pretrained weight is updated.",
        "",
        "## What loops here",
        "",
        "`L_level` is two post-norm attention/SwiGLU blocks, shared by `z_L` and `z_H`. Each H cycle applies that module 4 times to update `z_L`, then once to update `z_H`. Embeddings, including the per-puzzle embedding, stay FP32. Every `CastedLinear` (the eight core projections plus the grid head and halt head) is quantized once and reused. Evaluation always runs 16 ACT steps: halting is disabled outside training, and there is no KV cache.",
        "",
        "H cycles are swept over 1, 2, 3, 4, and 6. The trained depth is 3. Grid labels use the official encoding (pad 0 ignored, EOS 1, colors 2–11). Cross-entropy is the training stablemax loss, averaged over supervised cells. Perplexity is `exp(CE)` on that 12-way vocabulary. Exact accuracy is the fraction of grids whose supervised cells are all correct. These are single unaugmented puzzles, not the augmented majority vote behind the public 40% / 6.2% ARC numbers.",
        "",
        "Quantization is asymmetric per-group round-to-nearest, group size 128. The forward pass multiplies dequantized FP32 weights. **No low-bit kernel is used**, so latency is not an integer-GEMM speedup.",
        "",
        "## Baseline",
        "",
        "| checkpoint | method | H | CE | PPL | exact acc | logits MSE | ms/example |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ctx["baseline"]:
        if int(row["h_cycles"]) not in (1, 3, 6):
            continue
        lines.append(
            f"| {row['checkpoint']} | {row['method']} | {row['h_cycles']} | {_fmt(row['ce'], 4)} | {_fmt(row['ppl'], 3)} | {_fmt(100 * float(row['exact_accuracy']), 1)}% | {_fmt(row['logits_mse'], 4)} | {_fmt(row['latency_ms_per_example'], 2)} |"
        )
    lines += ["", "## Error across H cycles", ""]
    lines.append("Plot files: `plots/error_vs_loop_v1.png` and `plots/error_vs_loop_v2.png`. z_H is the state read by the grid head. Local error is one quantized H cycle on the FP state; propagated error is the rest.")
    lines.append("")
    lines.append("| checkpoint | bits | H | RMS | relative | cosine vs previous | local/total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in ctx["errors"]:
        if int(row["h_cycle"]) not in (1, 3, 6):
            continue
        lines.append(
            f"| {row['checkpoint']} | {row['bits']} | {row['h_cycle']} | {_fmt(row['abs_rms'], 3)} | {_fmt(row['rel_rms'], 3)} | {_fmt(row['cos_with_prev'], 3)} | {_fmt(row['local_over_total_rms'], 3)} |"
        )
    lines += ["", "## Oracle", "", "At 3 H cycles, alpha 0 and 1:", ""]
    lines.append("| checkpoint | bits | mode | alpha | CE | exact acc | CE gap closed |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for row in ctx["oracle"]:
        if int(row["h_cycles"]) != 3 or float(row["alpha"]) not in (0.0, 1.0):
            continue
        lines.append(
            f"| {row['checkpoint']} | {row['bits']} | {row['mode']} | {_fmt(row['alpha'], 2)} | {_fmt(row['ce'], 4)} | {_fmt(100 * float(row['exact_accuracy']), 1)}% | {_fmt(100 * float(row['ce_gap_closed']), 1)}% |"
        )
    lines += ["", ctx["reason"], "", "## Linear corrector", ""]
    if not ctx["ran_exp4"]:
        lines.append("Not fit. " + ctx["reason"])
    else:
        lines.append("Fit only on calibration demonstrations. Test grids follow the corrected quantized trajectory. The map is `U @ V.T @ state`, separate for `z_H` and `z_L`.")
        lines.append("")
        lines.append("| checkpoint | bits | H | method | rank | CE | exact acc | gap closed | pred cosine | extra params | latency ratio |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in ctx["predictor"]:
            if row.get("status") != "ok" or int(row["h_cycles"]) != 3:
                continue
            if row["rank"] not in ("", 16, "16") and row["method"] not in ("none", "oracle"):
                continue
            rank = row["rank"] if row["rank"] != "" else "-"
            lines.append(
                f"| {row['checkpoint']} | {row['bits']} | {row['h_cycles']} | {row['method']} | {rank} | {_fmt(row['ce'], 4)} | {_fmt(100 * float(row['exact_accuracy']), 1)}% | {_fmt(100 * float(row['ce_gap_closed']), 1)}% | {_fmt(row['pred_cosine'], 3)} | {row['n_extra_params']} | {_fmt(row['latency_ratio'], 3)} |"
            )
    lines += ["", "## Is it promising here?", "", ctx["verdict"], ""]
    path.write_text("\n".join(lines))
    print(f"wrote {path}", flush=True)


def build_verdict(errors, oracle, predictor, ran_exp4) -> str:
    cross = [r for r in oracle if float(r["alpha"]) == 1.0 and int(r["n_subsequent_h_cycles"]) >= 1 and r["mode"] == "every_except_final"]
    if not cross:
        return "The oracle produced no cross-loop comparison."
    best = max(cross, key=lambda r: r["ce_gap_closed"] if not math.isnan(float(r["ce_gap_closed"])) else -1e9)
    text = (
        f"The strongest full-state correction before later H cycles is {best['checkpoint']} INT{best['bits']} "
        f"at {best['h_cycles']} H cycles, closing {float(best['ce_gap_closed']):.1%} of the stablemax CE gap "
        f"and moving exact accuracy by {float(best['exact_accuracy_gain']):+.3f}."
    )
    if float(best["ce_gap_closed"]) < 0.10:
        return text + " That is not enough recovery to justify a learned cross-loop corrector on this model."
    if not ran_exp4:
        return text + " The linear corrector was not fit."
    learned = [r for r in predictor if r.get("method") in ("shared", "independent") and r.get("status") == "ok"]
    if not learned:
        return text
    top = max(learned, key=lambda r: float(r["ce_gap_closed"]) if r["ce_gap_closed"] != "" and not math.isnan(float(r["ce_gap_closed"])) else -1e9)
    text += (
        f" The best rank-{top['rank']} {top['method']} map ({top['checkpoint']} INT{top['bits']}, {top['h_cycles']} H cycles) "
        f"closes {float(top['ce_gap_closed']):.1%} of the CE gap, {float(top['fraction_of_oracle_recovery']):.0%} of the oracle's recovery, "
        f"with {int(top['n_extra_params'])} extra FP32 parameters."
    )
    if float(top["fraction_of_oracle_recovery"]) >= 0.3 and float(top["ce_gap_closed"]) >= 0.1:
        return text + " Cross-loop linear feedback is promising enough to study further, still as fake quantization rather than a faster kernel."
    if float(top["fraction_of_oracle_recovery"]) >= 0.1:
        return text + " The oracle has real headroom and the linear map captures only a slice of it."
    return text + " The oracle can move the loss, but this low-rank map does not predict the carried error well enough."


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This run needs a CUDA GPU.")
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    device = torch.device("cuda")
    out_dir = ROOT / "arc_results"
    all_base, all_err, all_oracle, all_pred = [], [], [], []
    reasons = []
    ran_any = False
    for variant in args.variants:
        base, err, oracle, pred, _ok, reason, ran = run_variant(variant, args, device, out_dir)
        all_base.extend(base)
        all_err.extend(err)
        all_oracle.extend(oracle)
        all_pred.extend(pred)
        reasons.append(f"{variant}: {reason}")
        ran_any = ran_any or ran
        write_csv(out_dir / "baseline_results.csv", all_base)
        write_csv(out_dir / "error_analysis.csv", all_err)
        write_csv(out_dir / "oracle_results.csv", all_oracle)
    write_csv(out_dir / "baseline_results.csv", all_base)
    write_csv(out_dir / "error_analysis.csv", all_err)
    write_csv(out_dir / "oracle_results.csv", all_oracle)
    # Predictor rows can mix a skip row with full rows. Keep one schema.
    if any(row.get("status") == "ok" for row in all_pred):
        all_pred = [row for row in all_pred if row.get("status") == "ok"]
    write_csv(out_dir / "predictor_results.csv", all_pred)
    reason = " ".join(reasons)
    verdict = build_verdict(all_err, all_oracle, all_pred, ran_any)
    write_report(out_dir / "report.md", {
        "baseline": all_base,
        "errors": all_err,
        "oracle": all_oracle,
        "predictor": all_pred,
        "reason": reason,
        "ran_exp4": ran_any,
        "verdict": verdict,
    })
    (out_dir / "summary.json").write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "low_bit_kernel_used": False,
        "ran_exp4": ran_any,
        "reason": reason,
        "verdict": verdict,
    }, indent=2))
    print(verdict, flush=True)


if __name__ == "__main__":
    main()
