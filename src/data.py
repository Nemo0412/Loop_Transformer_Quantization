"""WikiText-2 token windows. Calibration is validation; evaluation is test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def _sha256_tensor(tokens: torch.Tensor) -> str:
    raw = tokens.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _parquet_path(root: Path, split: str) -> Path:
    """WikiText-2 raw, fetched as parquet so the run does not depend on the old dataset script."""
    path = root / "data" / "wikitext" / f"wikitext-2-raw-v1-{split}.parquet"
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
    )
    import urllib.request

    print(f"downloading {url}", flush=True)
    urllib.request.urlretrieve(url, path)
    return path


def _load_split_text(root: Path, split: str) -> str:
    import pyarrow.parquet as pq

    table = pq.read_table(_parquet_path(root, split), columns=["text"])
    return "\n\n".join(table.column("text").to_pylist())


def _tokenize(tokenizer_path: Path, text: str) -> torch.Tensor:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ids = tokenizer.encode(text).ids
    tokens = torch.tensor(ids, dtype=torch.long)
    if tokens.numel() == 0:
        raise RuntimeError("tokenizer produced no tokens")
    if int(tokens.min()) < 0 or int(tokens.max()) >= 16384:
        raise RuntimeError("token id outside the TRM-LM vocabulary")
    return tokens


def _windows(tokens: torch.Tensor, seq_len: int) -> torch.Tensor:
    n = tokens.numel() // seq_len
    if n < 1:
        raise RuntimeError(f"not enough tokens ({tokens.numel()}) for seq_len={seq_len}")
    return tokens[: n * seq_len].contiguous().view(n, seq_len)


def build_or_load_tokens(
    root: Path,
    tokenizer_path: Path,
    seq_len: int,
    calib_seqs: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "tokens_meta.json"
    eval_path = data_dir / "eval_tokens.pt"
    calib_path = data_dir / "calib_tokens.pt"

    if meta_path.exists() and eval_path.exists() and calib_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("seq_len") == seq_len and meta.get("seed") == seed and meta.get("calib_seqs_requested") == calib_seqs:
            eval_tokens = torch.load(eval_path, weights_only=True)
            calib_tokens = torch.load(calib_path, weights_only=True)
            if _sha256_tensor(eval_tokens) != meta["eval_sha256"]:
                raise RuntimeError("eval token checksum mismatch")
            return eval_tokens, calib_tokens, meta

    eval_tokens = _windows(_tokenize(tokenizer_path, _load_split_text(root, "test")), seq_len)
    valid_tokens = _windows(_tokenize(tokenizer_path, _load_split_text(root, "validation")), seq_len)
    n_calib = min(calib_seqs, valid_tokens.shape[0])
    calib_tokens = valid_tokens[:n_calib].contiguous()

    meta = {
        "dataset": "wikitext-2-raw-v1",
        "eval_split": "test",
        "calib_split": "validation",
        "seq_len": seq_len,
        "seed": seed,
        "calib_seqs_requested": calib_seqs,
        "n_eval_seq": int(eval_tokens.shape[0]),
        "n_calib_seq": int(calib_tokens.shape[0]),
        "n_eval_tokens": int(eval_tokens.numel()),
        "n_calib_tokens": int(calib_tokens.numel()),
        "n_eval_predictions": int(eval_tokens.shape[0] * (seq_len - 1)),
        "eval_sha256": _sha256_tensor(eval_tokens),
        "calib_sha256": _sha256_tensor(calib_tokens),
        "disjoint_splits": True,
    }
    torch.save(eval_tokens, eval_path)
    torch.save(calib_tokens, calib_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    return eval_tokens, calib_tokens, meta
