"""Loop-explicit forward for TRM-LM v2.

The pretrained forward is:

    embed -> prelude (1 block) -> [gain/tag mix + core (5 blocks)] * r -> coda -> tied head

Hidden states used throughout the experiments are the activations at the
boundaries of the repeated five-block core: after each core pass, before the
next loop's gain/tag mix. The coda reads the final boundary state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from modeling_trm2 import rms, rope_tables


def loop_body(model, h, residual, loop_index, cos, sin):
    """One core loop. `residual` is the prelude output re-injected every loop."""
    j = min(loop_index, model.c["MAX_ITERS"] - 1)
    y = rms(h) * model.gain[j] + residual + model.tag[j]
    for block in model.core:
        y = block(y, cos, sin)
    return y


def prelude(model, ids):
    cos, sin = rope_tables(ids.shape[1], model.c["D"] // model.c["HEADS"], ids.device)
    x = model.embed(ids)
    for block in model.prelude:
        x = block(x, cos, sin)
    return x, cos, sin


def run_core(model, h0, residual, n_loops, cos, sin):
    h = h0
    states = []
    for i in range(n_loops):
        h = loop_body(model, h, residual, i, cos, sin)
        states.append(h)
    return states


def logits_from_state(model, h, cos, sin):
    y = h
    for block in model.coda:
        y = block(y, cos, sin)
    return (rms(y) * model.nf) @ model.embed.weight.t()


def forward_logits(model, ids, n_loops):
    e, cos, sin = prelude(model, ids)
    states = run_core(model, e, e, n_loops, cos, sin)
    return logits_from_state(model, states[-1], cos, sin)


def should_correct(mode: str, loop_index: int, n_loops: int) -> bool:
    """loop_index is 0-based. Loop 1 in the writeup is index 0."""
    if mode == "after_loop_1":
        return loop_index == 0
    if mode == "after_loop_2":
        return loop_index == 1
    if mode == "every_except_final":
        return loop_index < n_loops - 1
    raise ValueError(mode)


def forward_with_state_correction(model, ids, n_loops, fp_states, mode: str, alpha: float):
    """Run quantized loops, optionally replacing a fraction of the live state error.

    The error is recomputed from the current quantized state and the FP state at
    the same loop index. Earlier corrections therefore change later targets.
    """
    e, cos, sin = prelude(model, ids)
    h = e
    post = []
    for i in range(n_loops):
        h = loop_body(model, h, e, i, cos, sin)
        if alpha != 0.0 and should_correct(mode, i, n_loops):
            h = h + alpha * (fp_states[i] - h)
        post.append(h)
    return logits_from_state(model, h, cos, sin), post


def forward_with_predictor(model, ids, n_loops, fp_states, predictors, collect_predictions: bool):
    """Add a low-rank error prediction after every non-final loop.

    `predictors` is a list of (U, V) pairs. A single shared pair may be passed
    once per loop. Correction is h <- h + U @ V.T @ h, with h a column vector.
    FP states are metric targets only; they are never written into the trajectory.
    """
    e, cos, sin = prelude(model, ids)
    h = e
    pred_sq = 0.0
    pred_dot = 0.0
    pred_pn = 0.0
    pred_tn = 0.0
    pred_count = 0
    cos_sum = 0.0
    cos_n = 0
    for i in range(n_loops):
        h = loop_body(model, h, e, i, cos, sin)
        if i < n_loops - 1:
            u, v = predictors[i]
            pred = (h @ v) @ u.T
            if collect_predictions:
                target = fp_states[i] - h
                diff = pred.float() - target.float()
                pred_sq += float(diff.pow(2).sum().item())
                pred_count += diff.numel()
                pf = pred.float().reshape(-1, pred.shape[-1])
                tf = target.float().reshape(-1, target.shape[-1])
                pred_dot += float((pf * tf).sum().item())
                pred_pn += float(pf.pow(2).sum().item())
                pred_tn += float(tf.pow(2).sum().item())
                token_cos = (pf * tf).sum(-1) / (
                    pf.norm(dim=-1).clamp(min=1e-8) * tf.norm(dim=-1).clamp(min=1e-8)
                )
                cos_sum += float(token_cos.sum().item())
                cos_n += int(token_cos.numel())
            h = h + pred
    logits = logits_from_state(model, h, cos, sin)
    stats = {
        "pred_sq": pred_sq,
        "pred_dot": pred_dot,
        "pred_pn": pred_pn,
        "pred_tn": pred_tn,
        "pred_count": pred_count,
        "cos_sum": cos_sum,
        "cos_n": cos_n,
    }
    return logits, stats


def token_ce_sum(logits, ids) -> tuple[float, int]:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = ids[:, 1:]
    nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="sum",
    )
    return float(nll.item()), int(shift_labels.numel())


def mse_sum(a, b) -> tuple[float, int]:
    diff = a.float() - b.float()
    return float(diff.pow(2).sum().item()), int(diff.numel())
