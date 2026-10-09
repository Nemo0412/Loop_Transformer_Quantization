#!/usr/bin/env python3
"""Tiny CPU smoke test for the Ouro-1.4B forward API.

Confirms: (1) model.model returns per-loop hidden_states_list, (2) overriding
total_ut_steps changes the number of loops, (3) lm_head applies per loop.
No GPU needed; runs on an 8-token toy input.
"""
import os
import sys
import torch

MODEL = "ByteDance/Ouro-1.4B"


def main():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    for k, v in (("pad_token_id", 0), ("bos_token_id", 0), ("eos_token_id", 0)):
        if not hasattr(cfg, k) or getattr(cfg, k) is None:
            setattr(cfg, k, v)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("config total_ut_steps =", model.config.total_ut_steps)
    print("inner  total_ut_steps =", model.model.total_ut_steps)

    T = 6
    model.config.total_ut_steps = T
    model.model.total_ut_steps = T

    ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids
    print("input ids shape", tuple(ids.shape))

    with torch.inference_mode():
        out = model.model(input_ids=ids, use_cache=False)

    # OuroModel.forward returns (BaseModelOutputWithPast, hidden_states_list, gate_list)
    if isinstance(out, tuple):
        base, hs_list, gate_list = out
    else:
        raise SystemExit(f"unexpected return type {type(out)}")

    print("num loops returned =", len(hs_list))
    assert len(hs_list) == T, (len(hs_list), T)
    for t, h in enumerate(hs_list):
        print(f"  loop {t}: hidden {tuple(h.shape)} dtype {h.dtype}")

    # per-loop logits via lm_head
    for t, h in enumerate(hs_list):
        logits = model.lm_head(h)
        last = logits[0, -1]
        print(f"  loop {t}: logits {tuple(logits.shape)} argmax={int(last.argmax())} "
              f"max={float(last.max()):.3f}")

    # list the Linear modules inside the shared core (decoder layers)
    n_lin = 0
    names = []
    for name, m in model.model.layers.named_modules():
        if isinstance(m, torch.nn.Linear):
            n_lin += 1
            if len(names) < 8:
                names.append(name + " " + str(tuple(m.weight.shape)))
    print("core Linear count =", n_lin)
    print("sample:", names)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
