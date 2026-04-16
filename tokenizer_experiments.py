#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from tests.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

SPECIAL_TOKENS = ["<|endoftext|>"]
SEP = "<|endoftext|>"

TINY_TOKENIZER_JSON = DATA_DIR / "tiny_vocab.json"
OWT_TOKENIZER_JSON = DATA_DIR / "owb_vocab.json"

TINY_TRAIN = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
TINY_DEV = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
OWT_TRAIN = DATA_DIR / "owt_train.txt"
OWT_DEV = DATA_DIR / "owt_valid.txt"


def load_tokenizer(which: str) -> Tokenizer:
    if which == "tiny":
        path = TINY_TOKENIZER_JSON
    elif which == "owt":
        path = OWT_TOKENIZER_JSON
    else:
        raise ValueError(f"unknown tokenizer: {which}")
    if not path.exists():
        raise FileNotFoundError(f"tokenizer json not found: {path}")
    return Tokenizer.from_files(str(path), str(path), special_tokens=SPECIAL_TOKENS)


def iter_documents(path: Path, sep: str = SEP, chunk_size: int = 1 << 20) -> Iterator[str]:
    buf = ""
    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(sep)
            for doc in parts[:-1]:
                if doc:
                    yield doc
            buf = parts[-1]
        if buf:
            yield buf


def reservoir_sample_docs(path: Path, k: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    sample: list[str] = []
    n = 0
    for doc in iter_documents(path):
        n += 1
        if len(sample) < k:
            sample.append(doc)
            continue
        j = rng.randrange(n)
        if j < k:
            sample[j] = doc
    if len(sample) < k:
        raise RuntimeError(f"only found {len(sample)} docs in {path}, expected >= {k}")
    return sample


def compression_ratio_bytes_per_token(tokenizer: Tokenizer, docs: list[str]) -> tuple[float, int, int]:
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))
    if total_tokens == 0:
        raise RuntimeError("total_tokens is 0")
    return total_bytes / total_tokens, total_bytes, total_tokens


def encode_docs_flat(tokenizer: Tokenizer, docs: list[str]) -> tuple[np.ndarray, list[int], int, int]:
    all_ids: list[int] = []
    doc_lens: list[int] = []
    total_bytes = 0
    for doc in docs:
        ids = tokenizer.encode(doc)
        doc_lens.append(len(ids))
        all_ids.extend(ids)
        total_bytes += len(doc.encode("utf-8"))
    arr = np.asarray(all_ids, dtype=np.uint16)
    return arr, doc_lens, total_bytes, int(arr.size)


def run_a() -> None:
    tiny_tok = load_tokenizer("tiny")
    owt_tok = load_tokenizer("owt")

    tiny_docs = reservoir_sample_docs(TINY_TRAIN, 10, seed=42)
    owt_docs = reservoir_sample_docs(OWT_TRAIN, 10, seed=42)

    tiny_on_tiny = compression_ratio_bytes_per_token(tiny_tok, tiny_docs)
    owt_on_owt = compression_ratio_bytes_per_token(owt_tok, owt_docs)

    print(
        f"[a] TinyStories tokenizer compression ratio (bytes/token): {tiny_on_tiny[0]:.6f}",
        file=sys.stderr,
    )
    print(
        f"[a] OpenWebText tokenizer compression ratio (bytes/token): {owt_on_owt[0]:.6f}",
        file=sys.stderr,
    )

    tiny_arr, tiny_doc_lens, tiny_total_bytes, tiny_total_tokens = encode_docs_flat(tiny_tok, tiny_docs)
    owt_arr, owt_doc_lens, owt_total_bytes, owt_total_tokens = encode_docs_flat(owt_tok, owt_docs)

    tiny_npy = DATA_DIR / "tokenizer_experiments.a.tiny_ids.npy"
    owt_npy = DATA_DIR / "tokenizer_experiments.a.owt_ids.npy"
    np.save(tiny_npy, tiny_arr)
    np.save(owt_npy, owt_arr)

    meta = {
        "tiny_train_sample_docs": 10,
        "owt_train_sample_docs": 10,
        "tiny_tokenizer_on_tiny": {
            "ids_npy": str(tiny_npy),
            "doc_token_lengths": tiny_doc_lens,
            "total_bytes": tiny_total_bytes,
            "total_tokens": tiny_total_tokens,
        },
        "owt_tokenizer_on_owt": {
            "ids_npy": str(owt_npy),
            "doc_token_lengths": owt_doc_lens,
            "total_bytes": owt_total_bytes,
            "total_tokens": owt_total_tokens,
        },
    }
    meta_path = DATA_DIR / "tokenizer_experiments.a.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote: {tiny_npy}")
    print(f"wrote: {owt_npy}")
    print(f"wrote: {meta_path}")


def run_b() -> None:
    tiny_tok = load_tokenizer("tiny")
    owt_tok = load_tokenizer("owt")

    owt_docs = reservoir_sample_docs(OWT_TRAIN, 10, seed=42)

    tiny_on_owt = compression_ratio_bytes_per_token(tiny_tok, owt_docs)
    owt_on_owt = compression_ratio_bytes_per_token(owt_tok, owt_docs)

    ratio_delta = tiny_on_owt[0] - owt_on_owt[0]
    print(
        f"[b] TinyStories tokenizer on OWT compression ratio (bytes/token): {tiny_on_owt[0]:.6f}",
        file=sys.stderr,
    )
    print(
        f"[b] OpenWebText tokenizer on OWT compression ratio (bytes/token): {owt_on_owt[0]:.6f}",
        file=sys.stderr,
    )
    print(f"[b] Compression ratio delta tiny-owt: {ratio_delta:.6f}", file=sys.stderr)

    tiny_arr, tiny_doc_lens, tiny_total_bytes, tiny_total_tokens = encode_docs_flat(tiny_tok, owt_docs)
    owt_arr, owt_doc_lens, owt_total_bytes, owt_total_tokens = encode_docs_flat(owt_tok, owt_docs)

    tiny_npy = DATA_DIR / "tokenizer_experiments.b.tiny_on_owt_ids.npy"
    owt_npy = DATA_DIR / "tokenizer_experiments.b.owt_on_owt_ids.npy"
    np.save(tiny_npy, tiny_arr)
    np.save(owt_npy, owt_arr)

    meta = {
        "owt_train_sample_docs": 10,
        "tiny_tokenizer_on_owt": {
            "ids_npy": str(tiny_npy),
            "doc_token_lengths": tiny_doc_lens,
            "total_bytes": tiny_total_bytes,
            "total_tokens": tiny_total_tokens,
        },
        "owt_tokenizer_on_owt": {
            "ids_npy": str(owt_npy),
            "doc_token_lengths": owt_doc_lens,
            "total_bytes": owt_total_bytes,
            "total_tokens": owt_total_tokens,
        },
        "qualitative": (
            "TinyStories tokenizer on OpenWebText usually yields worse compression "
            "because its merges are specialized for TinyStories distribution."
        ),
    }
    meta_path = DATA_DIR / "tokenizer_experiments.b.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote: {tiny_npy}")
    print(f"wrote: {owt_npy}")
    print(f"wrote: {meta_path}")


def read_prefix_text(path: Path, target_bytes: int) -> tuple[str, int]:
    with path.open("rb") as f:
        bs = f.read(target_bytes)
    text = bs.decode("utf-8", errors="replace")
    actual_bytes = len(text.encode("utf-8"))
    return text, actual_bytes


def benchmark_throughput(tokenizer: Tokenizer, path: Path, target_bytes: int = 16 * 1024 * 1024) -> dict:
    text, actual_bytes = read_prefix_text(path, target_bytes)
    t0 = time.perf_counter()
    ids = tokenizer.encode(text)
    dt = time.perf_counter() - t0
    if dt <= 0:
        raise RuntimeError("non-positive benchmark time")
    return {
        "benchmark_bytes": actual_bytes,
        "benchmark_tokens": len(ids),
        "seconds": dt,
        "bytes_per_second": actual_bytes / dt,
        "tokens_per_second": len(ids) / dt,
    }


def run_c() -> None:
    tiny_tok = load_tokenizer("tiny")
    owt_tok = load_tokenizer("owt")

    tiny_perf = benchmark_throughput(tiny_tok, TINY_TRAIN)
    owt_perf = benchmark_throughput(owt_tok, OWT_TRAIN)

    pile_bytes = 825_000_000_000  # 825 GB (decimal)

    tiny_seconds = pile_bytes / tiny_perf["bytes_per_second"]
    owt_seconds = pile_bytes / owt_perf["bytes_per_second"]

    out = {
        "assumption": "Pile size = 825,000,000,000 bytes (825 GB decimal)",
        "benchmark_note": "throughput estimated using 16 MiB prefix per corpus (faster local benchmark)",
        "tiny_tokenizer_throughput": tiny_perf,
        "owt_tokenizer_throughput": owt_perf,
        "estimated_time_for_pile": {
            "tiny_seconds": tiny_seconds,
            "tiny_hours": tiny_seconds / 3600,
            "owt_seconds": owt_seconds,
            "owt_hours": owt_seconds / 3600,
        },
    }

    out_path = DATA_DIR / "tokenizer_experiments.c.txt"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote: {out_path}")


def encode_file_to_npy(tokenizer: Tokenizer, in_path: Path, out_npy: Path, flush_every: int = 1_000_000) -> dict:
    tmp_raw = out_npy.with_suffix(out_npy.suffix + ".raw")

    total = 0
    max_id = 0
    buf: list[int] = []

    with tmp_raw.open("wb") as raw, in_path.open("r", encoding="utf-8") as f:
        for tid in tokenizer.encode_iterable(f):
            if tid > 65535:
                raise ValueError(f"token id {tid} exceeds uint16 range")
            if tid > max_id:
                max_id = tid
            buf.append(tid)
            total += 1
            if len(buf) >= flush_every:
                np.asarray(buf, dtype=np.uint16).tofile(raw)
                buf.clear()
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(raw)
            buf.clear()

    mmap = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.uint16, shape=(total,))
    raw_arr = np.memmap(tmp_raw, mode="r", dtype=np.uint16, shape=(total,))
    mmap[:] = raw_arr[:]
    mmap.flush()
    del raw_arr
    del mmap
    tmp_raw.unlink(missing_ok=True)

    return {
        "input": str(in_path),
        "output": str(out_npy),
        "num_tokens": total,
        "max_token_id": max_id,
        "dtype": "uint16",
        "nbytes": int(total * np.dtype(np.uint16).itemsize),
    }


def run_d() -> None:
    tiny_tok = load_tokenizer("tiny")
    owt_tok = load_tokenizer("owt")

    outputs = {
        "tinystories_train": DATA_DIR / "tinystories_train_ids.npy",
        "tinystories_dev": DATA_DIR / "tinystories_dev_ids.npy",
        "openwebtext_train": DATA_DIR / "openwebtext_train_ids.npy",
        "openwebtext_dev": DATA_DIR / "openwebtext_dev_ids.npy",
    }

    meta = {
        "tinystories_train": encode_file_to_npy(tiny_tok, TINY_TRAIN, outputs["tinystories_train"]),
        "tinystories_dev": encode_file_to_npy(tiny_tok, TINY_DEV, outputs["tinystories_dev"]),
        "openwebtext_train": encode_file_to_npy(owt_tok, OWT_TRAIN, outputs["openwebtext_train"]),
        "openwebtext_dev": encode_file_to_npy(owt_tok, OWT_DEV, outputs["openwebtext_dev"]),
        "why_uint16": (
            "Both vocab sizes here (10K and 32K) are below 65,536, so every token id fits in uint16, "
            "which halves storage versus uint32."
        ),
    }

    meta_path = DATA_DIR / "tokenizer_experiments.d.metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote metadata: {meta_path}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenizer experiments for parts a/b/c/d")
    parser.add_argument("mode", choices=["a", "b", "c", "d"], help="Which experiment to run")
    args = parser.parse_args()

    if args.mode == "a":
        run_a()
    elif args.mode == "b":
        run_b()
    elif args.mode == "c":
        run_c()
    elif args.mode == "d":
        run_d()


if __name__ == "__main__":
    main()
