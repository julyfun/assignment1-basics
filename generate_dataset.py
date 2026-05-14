#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tests.tokenizer import Tokenizer

SPECIAL_TOKENS = ["<|endoftext|>"]


def main():
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: python {sys.argv[0]} TOKENIZER_JSON INPUT_TXT OUTPUT_NPY")

    tok_path, in_path, out_path = map(Path, sys.argv[1:])
    tok = Tokenizer.from_files(str(tok_path), str(tok_path), special_tokens=SPECIAL_TOKENS)
    token_ids = tok.encode_file_u16(str(in_path))
    arr = np.asarray(token_ids, dtype=np.uint16)
    np.save(out_path, arr, allow_pickle=False)

    print(f"wrote {out_path}")
    print(f"tokens: {arr.size:,}")
    print("dtype: uint16")


if __name__ == "__main__":
    main()
