# scripts/prepare_labels.py
import json
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--taxonomy", default="taxonomy.json")
parser.add_argument("--out", default="labels.csv")
parser.add_argument("--add-variants", action="store_true", help="Add textual variants to improve CLIP")
args = parser.parse_args()

with open(args.taxonomy, "r", encoding="utf-8") as f:
    tax = json.load(f)

labels = []
for cat, subs in tax.items():
    for sub in subs:
        base = f"{cat} > {sub}"
        labels.append(base)
        if args.add_variants:
            variants = [
                f"photo of {sub}",
                f"{sub} product",
                f"{cat} {sub}",
                f"{sub} - {cat}"
            ]
            for v in variants:
                labels.append(f"{cat} > {v}")
                
# dedupe and write
labels = list(dict.fromkeys(labels))
with open(args.out, "w", newline="", encoding="utf-8") as fout:
    writer = csv.writer(fout)
    for l in labels:
        writer.writerow([l])
print(f"Wrote {len(labels)} labels to {args.out}")
