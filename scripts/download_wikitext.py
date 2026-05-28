"""Download WikiText-103 (v1, pre-tokenised) from HuggingFace and write
the three token files that fairseq preprocess expects."""
from datasets import load_dataset

OUT = "/content/wt103-raw/wikitext-103"
SPLITS = [
    ("train",      "wiki.train.tokens"),
    ("validation", "wiki.valid.tokens"),
    ("test",       "wiki.test.tokens"),
]

print("Loading wikitext-103-v1 from HuggingFace ...")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

for split, fname in SPLITS:
    path = f"{OUT}/{fname}"
    with open(path, "w", encoding="utf-8") as f:
        for item in ds[split]:
            f.write(item["text"] + "\n")
    print(f"Wrote {path}")

print("Download complete.")
