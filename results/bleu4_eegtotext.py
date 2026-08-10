"""BLEU-4 per-pair analysis on EEGTOTEXT.txt — top2 and bottom2."""
import re
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

PATH = Path(__file__).parent / "EEGTOTEXT.txt"
text = PATH.read_text(encoding="utf-8", errors="replace").lstrip("﻿")

# Split on the 50-= separator
chunks = [c.strip() for c in text.split("=" * 50) if c.strip()]

pairs = []
for c in chunks:
    # Each chunk: "Predicted: <pred> True: <ref>"
    m = re.match(r"Predicted:\s*(.*?)\s*True:\s*(.*)", c, re.DOTALL)
    if not m:
        continue
    pred = m.group(1).strip()
    ref = m.group(2).strip()
    if pred and ref:
        pairs.append((pred, ref))

print(f"[INFO] parsed {len(pairs)} pairs")

sm = SmoothingFunction().method1
scored = []
for i, (pred, ref) in enumerate(pairs):
    ref_tokens = [ref.lower().split()]
    pred_tokens = pred.lower().split()
    if len(pred_tokens) == 0:
        continue
    score = sentence_bleu(ref_tokens, pred_tokens,
                          weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=sm)
    scored.append((i, score, pred, ref))

scored.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 70)
print("TOP 2 (highest BLEU-4)")
print("=" * 70)
for rank, (i, s, p, r) in enumerate(scored[:2], 1):
    print(f"\n[#{rank}] idx={i}  BLEU-4 = {s:.4f}")
    print(f"  Pred: {p}")
    print(f"  True: {r}")

print("\n" + "=" * 70)
print("BOTTOM 2 (lowest BLEU-4)")
print("=" * 70)
for rank, (i, s, p, r) in enumerate(scored[-2:][::-1], 1):
    print(f"\n[#{rank}] idx={i}  BLEU-4 = {s:.4f}")
    print(f"  Pred: {p}")
    print(f"  True: {r}")

# Corpus-level stat too
mean_bleu = sum(s for _, s, _, _ in scored) / len(scored)
print(f"\n[STAT] mean BLEU-4 over {len(scored)} pairs = {mean_bleu:.4f}")
