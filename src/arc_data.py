"""Rebuild ARC puzzle ids with the official TRM augmentation order.

Identifiers have to match the checkpoint's puzzle embedding rows. The official
builder draws 1000 augmentations per puzzle under numpy seed 42. This copy
keeps that RNG path and stores only the unaugmented train/eval grids.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ARC_MAX = 30
AUG_COUNT = 1000
AUG_RETRIES = 5
SEPARATOR = "|||"
IGNORE_LABEL = 0


def _dihedral(arr: np.ndarray, tid: int) -> np.ndarray:
    if tid == 0:
        return arr
    if tid == 1:
        return np.rot90(arr, k=1)
    if tid == 2:
        return np.rot90(arr, k=2)
    if tid == 3:
        return np.rot90(arr, k=3)
    if tid == 4:
        return np.fliplr(arr)
    if tid == 5:
        return np.flipud(arr)
    if tid == 6:
        return arr.T
    if tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    return arr


def _grid_hash(grid: np.ndarray) -> str:
    buffer = [int(x).to_bytes(1, byteorder="big") for x in grid.shape]
    buffer.append(grid.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def _puzzle_hash(examples: list[tuple[np.ndarray, np.ndarray]]) -> str:
    hashes = [f"{_grid_hash(inp)}|{_grid_hash(out)}" for inp, out in examples]
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def _encode_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = []
    for grid in (inp, out):
        nrow, ncol = grid.shape
        canvas = np.pad(
            grid + 2,
            ((0, ARC_MAX - nrow), (0, ARC_MAX - ncol)),
            constant_values=0,
        )
        if nrow < ARC_MAX:
            canvas[nrow, 0:ncol] = 1
        if ncol < ARC_MAX:
            canvas[0:nrow, ncol] = 1
        result.append(canvas.reshape(-1).astype(np.int16))
    return result[0], result[1]


def _load_subset(raw_dir: Path, name: str) -> list[tuple[str, dict]]:
    puzzles = json.loads((raw_dir / f"arc-agi_{name}_challenges.json").read_text())
    solutions_path = raw_dir / f"arc-agi_{name}_solutions.json"
    if solutions_path.exists():
        solutions = json.loads(solutions_path.read_text())
        for puzzle_id, grids in solutions.items():
            for idx, grid in enumerate(grids):
                puzzles[puzzle_id]["test"][idx]["output"] = grid
    else:
        for puzzle in puzzles.values():
            for example in puzzle["test"]:
                example.setdefault("output", [[0]])
    items = list(puzzles.items())
    np.random.shuffle(items)
    return items


def _as_u8(grid) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[0] > ARC_MAX or arr.shape[1] > ARC_MAX:
        raise RuntimeError(f"bad grid shape {arr.shape}")
    return arr


def build_variant(root: Path, variant: str, raw_dir: Path) -> dict:
    """variant is 'v1' or 'v2'."""
    if variant == "v1":
        subsets = ("training", "evaluation", "concept")
        train_name, test_name = "training", "evaluation"
    elif variant == "v2":
        subsets = ("training2", "evaluation2", "concept")
        train_name, test_name = "training2", "evaluation2"
    else:
        raise ValueError(variant)

    out_dir = root / "data" / f"arc_{variant}"
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("aug_count") == AUG_COUNT and meta.get("variant") == variant:
            return {
                "eval_inputs": np.load(out_dir / "eval_inputs.npy"),
                "eval_labels": np.load(out_dir / "eval_labels.npy"),
                "eval_ids": np.load(out_dir / "eval_ids.npy"),
                "calib_inputs": np.load(out_dir / "calib_inputs.npy"),
                "calib_labels": np.load(out_dir / "calib_labels.npy"),
                "calib_ids": np.load(out_dir / "calib_ids.npy"),
                "meta": meta,
            }

    np.random.seed(42)
    next_id = 1
    eval_inputs, eval_labels, eval_ids = [], [], []
    calib_inputs, calib_labels, calib_ids = [], [], []
    n_puzzles = 0

    for subset in subsets:
        print(f"encoding {variant} subset {subset}", flush=True)
        for name, puzzle in _load_subset(raw_dir, subset):
            train_examples = [(_as_u8(ex["input"]), _as_u8(ex["output"])) for ex in puzzle["train"]]
            test_examples = [(_as_u8(ex["input"]), _as_u8(ex["output"])) for ex in puzzle["test"]]
            by_dest = {"train": train_examples, "test": test_examples}
            original_hash = _puzzle_hash(train_examples + test_examples)
            kept = 1
            seen = {original_hash}
            original_id = next_id
            next_id += 1

            for _trial in range(AUG_RETRIES * AUG_COUNT):
                trans_id = int(np.random.randint(0, 8))
                mapping = np.concatenate(
                    [np.arange(0, 1, dtype=np.uint8), np.random.permutation(np.arange(1, 10, dtype=np.uint8))]
                )

                def _map(grid, mapping=mapping, trans_id=trans_id):
                    return _dihedral(mapping[grid], trans_id)

                aug_pairs = [(_map(inp), _map(out)) for dest in ("train", "test") for inp, out in by_dest[dest]]
                digest = _puzzle_hash(aug_pairs)
                if digest not in seen:
                    seen.add(digest)
                    kept += 1
                    next_id += 1
                if kept >= AUG_COUNT + 1:
                    break

            if subset == test_name:
                for inp, out in test_examples:
                    enc_in, enc_out = _encode_pair(inp, out)
                    eval_inputs.append(enc_in)
                    eval_labels.append(enc_out)
                    eval_ids.append(original_id)
            elif subset == train_name:
                for inp, out in train_examples:
                    enc_in, enc_out = _encode_pair(inp, out)
                    calib_inputs.append(enc_in)
                    calib_labels.append(enc_out)
                    calib_ids.append(original_id)
            n_puzzles += 1
            if n_puzzles % 100 == 0:
                print(f"  {variant} puzzles {n_puzzles} next_id {next_id}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "eval_inputs": np.stack(eval_inputs),
        "eval_labels": np.stack(eval_labels),
        "eval_ids": np.asarray(eval_ids, dtype=np.int32),
        "calib_inputs": np.stack(calib_inputs),
        "calib_labels": np.stack(calib_labels),
        "calib_ids": np.asarray(calib_ids, dtype=np.int32),
    }
    for key, value in arrays.items():
        np.save(out_dir / f"{key}.npy", value)
    meta = {
        "variant": variant,
        "aug_count": AUG_COUNT,
        "num_puzzle_identifiers": next_id,
        "n_eval": int(arrays["eval_inputs"].shape[0]),
        "n_calib": int(arrays["calib_inputs"].shape[0]),
        "n_puzzles": n_puzzles,
        "seq_len": ARC_MAX * ARC_MAX,
        "ignore_label_id": IGNORE_LABEL,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    arrays["meta"] = meta
    print(f"built {variant}: eval {meta['n_eval']} calib {meta['n_calib']} ids {next_id}", flush=True)
    return arrays
