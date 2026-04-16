from collections.abc import Iterable, Iterator
import json
from pathlib import Path
import regex

from cs336_basics import cs336_basics

PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self._inner = cs336_basics.Tokenizer(vocab, merges, special_tokens)
        self._pretoken_re = regex.compile(PAT)
        self._special_tokens = sorted((special_tokens or []), key=len, reverse=True)
        self._max_special_len = max((len(x) for x in (special_tokens or [])), default=0)
        
    @classmethod
    def from_files(
       cls, vocab_filepath, merges_filepath, special_tokens=None 
    ):
        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)
        if vocab_path.is_file() and merges_path.is_file():
            if vocab_path == merges_path:
                with open(vocab_path, encoding="utf-8") as f:
                    c = json.load(f)
                vocab = {int(k): bytes(v) for k, v in c["vocab"].items()}
                merges = [(bytes(a), bytes(b)) for a, b in c["merges"]]
                return cls(vocab, merges, special_tokens)
            return cls(
                *cls._load_vocab_merges(vocab_path, merges_path),
                special_tokens=special_tokens,
            )

        root_data = Path(__file__).resolve().parents[1] / "data" / vocab_filepath
        with open(root_data, encoding="utf-8") as f:
            c = json.load(f)
        vocab = {int(k): bytes(v) for k, v in c["vocab"].items()}
        merges = [(bytes(a), bytes(b)) for a, b in c["merges"]]
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _load_vocab_merges(vocab_path: Path, merges_path: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        with open(vocab_path, encoding="utf-8") as f:
            vocab_json = json.load(f)
        with open(merges_path, encoding="utf-8") as f:
            merges_json = json.load(f)
        vocab = {int(k): bytes(v) for k, v in vocab_json.items()}
        merges = [(bytes(a), bytes(b)) for a, b in merges_json]
        return vocab, merges
        
    def encode(self, text: str) -> list[int]:
        return self._inner.encode(text)
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        carry = ""
        for chunk in iterable:
            carry += chunk
            if not carry:
                continue

            cut_end = 0
            for m in self._pretoken_re.finditer(carry):
                s, e = m.span()
                if s < cut_end:
                    continue
                if e < len(carry):
                    cut_end = e

            # Avoid cutting through a potential special token crossing boundary.
            if self._max_special_len > 0 and cut_end > 0:
                changed = True
                while changed and cut_end > 0:
                    changed = False
                    start = max(0, cut_end - self._max_special_len + 1)
                    window = carry[start : min(len(carry), cut_end + self._max_special_len - 1)]
                    abs_start = start
                    for token in self._special_tokens:
                        idx = 0
                        while True:
                            j = window.find(token, idx)
                            if j < 0:
                                break
                            s = abs_start + j
                            e = s + len(token)
                            if s < cut_end < e:
                                cut_end = s
                                changed = True
                                break
                            idx = j + 1
                        if changed:
                            break

            if cut_end > 0:
                for token_id in self._inner.encode(carry[:cut_end]):
                    yield token_id
                carry = carry[cut_end:]

        if carry:
            for token_id in self._inner.encode(carry):
                yield token_id
        
    def decode(self, ids: list[int]) -> str:
        return self._inner.decode(ids)
        
