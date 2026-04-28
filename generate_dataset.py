#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tests.tokenizer import Tokenizer

SPECIAL_TOKENS = ["<|endoftext|>"]
SEP = "<|endoftext|>"


def main():
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: python {sys.argv[0]} TOKENIZER_JSON INPUT_TXT OUTPUT_NPY")

    tok_path, in_path, out_path = map(Path, sys.argv[1:])
    tok = Tokenizer.from_files(str(tok_path), str(tok_path), special_tokens=SPECIAL_TOKENS)

    tmp_raw = out_path.with_suffix(out_path.suffix + ".raw")
    total = 0
    buf = []

    with tmp_raw.open("wb") as raw, in_path.open("r", encoding="utf-8") as f:
        pbar = tqdm(unit="tok", unit_scale=True, desc="encoding")
        for tid in tok.encode_iterable(f):
            if tid > 65535:
                raise ValueError(f"token id {tid} exceeds uint16 range")

            buf.append(tid)
            total += 1
            pbar.update(1)

            if len(buf) >= 1_000_000:
                np.asarray(buf, dtype=np.uint16).tofile(raw)
                buf.clear()
                pbar.set_postfix(written=f"{total:,}")

        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(raw)

        pbar.set_postfix(written=f"{total:,}")
        pbar.close()

    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint16, shape=(total,))
    raw_arr = np.memmap(tmp_raw, mode="r", dtype=np.uint16, shape=(total,))
    arr[:] = raw_arr[:]
    arr.flush()

    del raw_arr, arr
    tmp_raw.unlink(missing_ok=True)

    print(f"wrote {out_path}")
    print(f"tokens: {total:,}")
    print("dtype: uint16")


if __name__ == "__main__":
    main()