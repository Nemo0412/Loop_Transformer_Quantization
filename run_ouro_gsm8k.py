#!/usr/bin/env python3
"""GSM8K accuracy of the Ouro-1.4B looped LM under weight-only quantization.

Evaluates exact-match accuracy on GSM8K (8-shot chain-of-thought, greedy decoding)
for the FP baseline and the standard weight-only quant grid INT8 / INT4 / FP8(E4M3),
optionally sweeping the Universal-Transformer loop depth (total_ut_steps).

Only the shared looped core (q/k/v/o/gate/up/down of all 24 decoder layers) is
quantized; embeddings, RMSNorm, lm_head and the exit gate stay FP. Weights are
dequantized to the compute dtype and a normal matmul is used (no low-bit kernel,
no speed claim). Generation runs in bf16 (native deployment precision) for speed.

Outputs land in ouro_results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from run_ouro_error import (
    MODEL_ID,
    QUANT_CONFIGS,
    load_ouro,
    quant_targets,
    quantize_in_place,
    restore_weights,
    stash_weights,
)

SEED = 42
GROUP_SIZE = 128
DEFAULT_CONFIGS = ("FP", "INT8", "INT4", "FP8")
GSM8K_URL = (
    "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/"
    "test-00000-of-00001.parquet"
)

# Standard 8-shot GSM8K chain-of-thought exemplars (Wei et al., 2022).
FEWSHOT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
     "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. "
     "So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
     "After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
     "How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
     "So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
     "How many toys does he have now?",
     "Shawn started with 5 toys. He then got 2 toys each from his mom and dad. "
     "So he got 2 * 2 = 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, "
     "from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
     "So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
     "How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
     "After losing 2 more, he had 35 - 2 = 33. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each is 5 * 3 = 15 dollars. "
     "So she has 23 - 15 = 8 dollars left. The answer is 8."),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", type=str, nargs="+", default=list(DEFAULT_CONFIGS))
    p.add_argument("--loops", type=int, nargs="+", default=[4],
                   help="UT depths to evaluate (Ouro trained at 4)")
    p.add_argument("--n-problems", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--group-size", type=int, default=GROUP_SIZE)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def patch_ouro_cache(model):
    """transformers>=4.55 makes Cache.key_cache a read-only property.

    Ouro's UniversalTransformerCache assigns self.key_cache = [] and then
    crashes. Give that class real setters backed by private lists, and swap
    the name the model looks up at runtime.
    """
    import sys

    mod = sys.modules[model.__class__.__module__]
    base = mod.UniversalTransformerCache

    class FixedUniversalTransformerCache(base):
        def __init__(self, max_cache_size=None):
            object.__setattr__(self, "_key_cache", [])
            object.__setattr__(self, "_value_cache", [])
            object.__setattr__(self, "layers", [])
            object.__setattr__(self, "_seen_tokens", 0)
            object.__setattr__(self, "max_cache_size", max_cache_size)

        @property
        def key_cache(self):
            return self._key_cache

        @key_cache.setter
        def key_cache(self, value):
            object.__setattr__(self, "_key_cache", value)

        @property
        def value_cache(self):
            return self._value_cache

        @value_cache.setter
        def value_cache(self, value):
            object.__setattr__(self, "_value_cache", value)

        def get_mask_sizes(self, cache_position, layer_idx=0):
            # Cache.get_mask_sizes indexes self.layers, which this class never
            # fills. All UT loops cache the same number of positions, so the
            # length from any populated index is the right one for the mask.
            past_seen = self.get_seq_length(0)
            return cache_position.shape[0] + past_seen, 0

    mod.UniversalTransformerCache = FixedUniversalTransformerCache


def gsm8k_path():
    path = ROOT / "data" / "gsm8k" / "test-00000-of-00001.parquet"
    if not (path.exists() and path.stat().st_size > 0):
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {GSM8K_URL}", flush=True)
        urllib.request.urlretrieve(GSM8K_URL, path)
    return path


def load_gsm8k(n):
    import pyarrow.parquet as pq

    table = pq.read_table(gsm8k_path(), columns=["question", "answer"])
    qs = table.column("question").to_pylist()
    ans = table.column("answer").to_pylist()
    gold = [extract_gold(a) for a in ans]
    items = list(zip(qs, gold))
    if n:
        items = items[:n]
    return items


def extract_gold(answer):
    # GSM8K gold answers end with "#### <number>"
    m = re.search(r"####\s*(.+)", answer)
    raw = m.group(1) if m else answer
    return normalize_num(raw)


def normalize_num(s):
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else None


def extract_pred(text):
    # prefer the pattern "The answer is X"; else last number before the next question
    text = text.split("Question:")[0]
    m = re.findall(r"[Tt]he answer is\s*([^\.\n]+)", text)
    if m:
        return normalize_num(m[-1])
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return normalize_num(nums[-1]) if nums else None


def build_prompt(question):
    parts = []
    for q, a in FEWSHOT:
        parts.append(f"Question: {q}\nAnswer: {a}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


@torch.inference_mode()
def eval_depth_config(model, tok, items, args, device):
    """Run greedy generation for the whole set; return accuracy and records."""
    correct = 0
    records = []
    bs = args.batch_size
    for start in range(0, len(items), bs):
        batch = items[start:start + bs]
        prompts = [build_prompt(q) for q, _ in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        # A growable DynamicCache keeps Ouro off its own legacy cache class
        # (needs_universal_cache returns False), so the modern mask utilities
        # work. Loop t caches at index t*num_layers + layer_idx.
        from transformers import DynamicCache

        out = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
            past_key_values=DynamicCache(),
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for (q, gold), text in zip(batch, texts):
            pred = extract_pred(text)
            ok = (pred is not None and gold is not None and pred == gold)
            correct += int(ok)
            records.append({"gold": gold, "pred": pred, "ok": ok})
        print(f"  [{start + len(batch)}/{len(items)}] running acc="
              f"{correct / (start + len(batch)):.4f}", flush=True)
    acc = correct / len(items) if items else float("nan")
    return acc, records


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_acc(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = []
    for c in DEFAULT_CONFIGS:
        if any(r["quant"] == c for r in rows) and c not in labels:
            labels.append(c)
    depths = sorted({r["loops"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if len(depths) > 1:
        for lab in labels:
            s = sorted([r for r in rows if r["quant"] == lab], key=lambda r: r["loops"])
            ax.plot([r["loops"] for r in s], [100 * r["accuracy"] for r in s],
                    marker="o", label=lab)
        ax.set_xlabel("loop depth (UT steps)")
    else:
        xs = range(len(labels))
        ax.bar(list(xs), [100 * next(r["accuracy"] for r in rows if r["quant"] == l) for l in labels])
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels)
        ax.set_xlabel("quantization")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title(f"Ouro-1.4B GSM8K accuracy ({depths if len(depths)>1 else 'T='+str(depths[0])})")
    ax.grid(True, alpha=0.3)
    if len(depths) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    torch.manual_seed(SEED)
    device = torch.device("cuda")

    if args.smoke:
        args.configs = ["FP", "INT4"]
        args.loops = [4]
        args.n_problems = 8
        args.batch_size = 4
        args.max_new_tokens = 200

    out_dir = ROOT / ("ouro_smoke" if args.smoke else "ouro_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    max_loops = max(args.loops)
    print(f"loading {MODEL_ID} (bf16, max loops={max_loops}) ...", flush=True)
    model, tok = load_ouro(device, max_loops, dtype=torch.bfloat16)
    patch_ouro_cache(model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    items = load_gsm8k(args.n_problems)
    print(f"GSM8K problems: {len(items)}", flush=True)

    targets = quant_targets(model, False)
    stash = stash_weights(targets)

    summary_rows = []
    for depth in args.loops:
        model.config.total_ut_steps = depth
        model.model.total_ut_steps = depth
        for label in args.configs:
            print(f"=== {label} @ T={depth} ===", flush=True)
            restore_weights(targets, stash)
            if label != "FP":
                quantize_in_place(targets, QUANT_CONFIGS[label], args.group_size)
            acc, records = eval_depth_config(model, tok, items, args, device)
            print(f"  {label} T={depth} accuracy={acc:.4f}", flush=True)
            summary_rows.append({
                "model": "ouro_1.4b",
                "quant": label,
                "loops": depth,
                "n_problems": len(items),
                "accuracy": acc,
                "n_correct": sum(r["ok"] for r in records),
            })
            write_csv(out_dir / "gsm8k_accuracy.csv", summary_rows)
    restore_weights(targets, stash)

    plot_acc(summary_rows, out_dir / "plots" / "ouro_gsm8k_accuracy.png")
    (out_dir / "gsm8k_summary.json").write_text(json.dumps({
        "model": MODEL_ID,
        "compute_dtype": "bfloat16",
        "group_size": args.group_size,
        "configs": args.configs,
        "loops": args.loops,
        "n_problems": len(items),
        "max_new_tokens": args.max_new_tokens,
        "prompt": "8-shot CoT, greedy",
        "rows": summary_rows,
    }, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    run()
