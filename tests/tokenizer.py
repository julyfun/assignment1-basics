from collections.abc import Iterable, Iterator
import json

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        
    @classmethod
    def from_files(
       cls, vocab_filepath, merges_filepath, special_tokens=None 
    ):
        assert vocab_filepath == merges_filepath
        with open(vocab_filepath, encoding="utf-8") as f:
            c = json.load(f)
            vocab, merges = c['vocab'], c['merges']
            return cls(
                vocab,
                merges,
                special_tokens,
            )
        
    def encode(self, text: str) -> list[int]:
        ...
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        ...
        
    def decode(self, ids: list[int]) -> str:
        ...
        