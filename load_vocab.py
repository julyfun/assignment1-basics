import json

with open("data/owb_vocab.json", encoding="utf-8") as f:
    c = json.load(f)
    vocab, merges = c['vocab'], c['merges']
    lent, long = max(
        (len(x), x)
        for x in vocab.values()
    )
    print(lent, str(bytes(long)))
    