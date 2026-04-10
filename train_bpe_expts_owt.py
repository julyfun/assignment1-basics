
import json

from tests.adapters import run_train_bpe

vocab, merges = run_train_bpe(
    input_path="data/owt_train.txt",
    vocab_size=32000,
    special_tokens=["<|endoftext|>"],
)

longest = max([(len(x), x) for x in vocab.values()])[1]
print("longest token:", longest)

json_vocab = {str(k): list(v) for k, v in vocab.items()}
json_merges = [[list(a), list(b)] for a, b in merges]

with open("data/owb_vocab.json", "w", encoding="utf-8") as f:
    json.dump(
        {"vocab": json_vocab, "merges": json_merges},
        f,
        indent=2,
        ensure_ascii=False,
    )

