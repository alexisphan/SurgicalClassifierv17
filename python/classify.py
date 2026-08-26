"""
ClinicalBERT Surgical Case Classifier
--------------------------------------
Trains Bio_ClinicalBERT on procedures_reference.csv,
classifies cases_to_classify.csv, writes:
  ../output/bert_classified.csv
  ../output/bert_classification_report.txt

Run from the python/ folder:
    python classify.py
"""

import os, csv, time
from datetime import datetime
from collections import defaultdict, Counter

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE, "data")
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROCEDURES_CSV = os.path.join(DATA_DIR, "procedures_reference.csv")
CASES_CSV      = os.path.join(DATA_DIR, "cases_to_classify.csv")
OUT_CSV        = os.path.join(OUTPUT_DIR, "bert_classified.csv")
OUT_REPORT     = os.path.join(OUTPUT_DIR, "bert_classification_report.txt")

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
NUM_EPOCHS = 40
BATCH_SIZE = 8
LR         = 2e-5
MAX_LEN    = 64

# ── Dataset ────────────────────────────────────────────────────────────────
class ProcedureDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=MAX_LEN
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


# ── Load training data ──────────────────────────────────────────────────────
print("[1/4] Loading training reference...")
train_df = pd.read_csv(PROCEDURES_CSV)
train_df.columns = [c.strip().strip('"') for c in train_df.columns]

# Identify text and label columns flexibly
name_col = [c for c in train_df.columns if "name" in c.lower()][0]
cat_col  = [c for c in train_df.columns if "cat" in c.lower()][0]

labels_list = sorted(train_df[cat_col].str.strip().unique().tolist())
label2id = {l: i for i, l in enumerate(labels_list)}
id2label = {i: l for l, i in label2id.items()}

train_texts  = train_df[name_col].str.strip().tolist()
train_labels = [label2id[l.strip()] for l in train_df[cat_col]]

print(f"      {len(train_texts)} training entries, {len(labels_list)} categories")

# ── Tokenize & train ────────────────────────────────────────────────────────
print("[2/4] Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels_list),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

train_dataset = ProcedureDataset(train_texts, train_labels, tokenizer)

args = TrainingArguments(
    output_dir=os.path.join(BASE, "python", "bert_output"),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    save_strategy="no",
    logging_steps=10,
    report_to="none",
)

print(f"      Fine-tuning for {NUM_EPOCHS} epochs...")
trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
trainer.train()

# ── Classify test cases ─────────────────────────────────────────────────────
print("[3/4] Classifying test cases...")
cases_df = pd.read_csv(CASES_CSV)
cases_df.columns = [c.strip().strip('"') for c in cases_df.columns]

opnames = (cases_df["opname"].str.strip() + " " + cases_df["dx"].str.strip()).tolist()

encodings = tokenizer(
    opnames, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt"
)

model.eval()
with torch.no_grad():
    outputs = model(**encodings)

probs = torch.softmax(outputs.logits, dim=1)
confidences, predicted_ids = probs.max(dim=1)

cases_df["ml_base_category"] = [id2label[i.item()] for i in predicted_ids]
cases_df["ml_confidence_pct"] = confidences.numpy().round(1) * 100

cases_df.to_csv(OUT_CSV, index=False)
print(f"      Saved: {OUT_CSV}")

# ── Generate report ─────────────────────────────────────────────────────────
print("[4/4] Generating classification report...")

total      = len(cases_df)
mean_conf  = cases_df["ml_confidence_pct"].mean()
low_conf   = cases_df[cases_df["ml_confidence_pct"] < 50]

# Accuracy vs ground truth optype (if column exists)
has_optype = "optype" in cases_df.columns
if has_optype:
    cases_df["_correct"] = (
        cases_df["ml_base_category"].str.strip() == cases_df["optype"].str.strip()
    )
    overall_acc = cases_df["_correct"].mean() * 100

# Prediction distribution
pred_dist = Counter(cases_df["ml_base_category"].str.strip())

# Per-category accuracy
cat_acc = {}
if has_optype:
    for cat, grp in cases_df.groupby("optype"):
        acc = (grp["ml_base_category"].str.strip() == grp["optype"].str.strip()).mean()
        cat_acc[cat] = (acc * 100, len(grp))

# Confidence distribution buckets
conf_buckets = {}
for lo, hi, label in [
    (0,  20,  "<20%"),
    (20, 30,  "20-30%"),
    (30, 40,  "30-40%"),
    (40, 50,  "40-50%"),
    (50, 60,  "50-60%"),
    (60, 70,  "60-70%"),
    (70, 80,  "70-80%"),
    (80, 90,  "80-90%"),
    (90, 101, "90-100%"),
]:
    n = ((cases_df["ml_confidence_pct"] >= lo) & (cases_df["ml_confidence_pct"] < hi)).sum()
    conf_buckets[label] = n

W = 80
SEP = "=" * W
SEP2 = "-" * 50

with open(OUT_REPORT, "w", encoding="utf-8") as f:

    def w(line=""): f.write(line + "\n")

    w(SEP)
    w("  ClinicalBERT SURGICAL CASE CLASSIFIER — CLASSIFICATION REPORT")
    w(f"  Model    : {MODEL_NAME}")
    w(f"  Epochs   : {NUM_EPOCHS}    Batch size : {BATCH_SIZE}    LR : {LR}")
    w(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(SEP)
    w()

    # ── Summary ──
    w("SUMMARY")
    w(SEP2)
    w(f"  Total cases classified      : {total:,}")
    w(f"  Training reference entries  : {len(train_texts):,}  ({len(labels_list)} categories)")
    w(f"  Average ML confidence       : {mean_conf:.1f}%")
    w(f"  Low confidence (<50%)       : {len(low_conf)}  ← review recommended")
    if has_optype:
        w(f"  Accuracy vs VitalDB optype  : {overall_acc:.1f}%  ({cases_df['_correct'].sum():,}/{total:,} correct)")
    w()

    # ── Prediction distribution ──
    w("PREDICTED CATEGORY DISTRIBUTION")
    w(SEP2)
    for cat, n in sorted(pred_dist.items(), key=lambda x: -x[1]):
        pct = n / total * 100
        w(f"  {cat:<35s}  {n:5d}  ({pct:5.1f}%)")
    w()

    # ── Accuracy by category ──
    if has_optype:
        w("ACCURACY VS VITALDB GROUND TRUTH (optype)")
        w(SEP2)
        w(f"  {'Category':<30s}  {'Accuracy':>8s}  {'Cases':>6s}  {'Correct':>7s}")
        w("  " + "-" * 58)
        for cat in sorted(cat_acc):
            acc_pct, n = cat_acc[cat]
            correct = round(acc_pct / 100 * n)
            w(f"  {cat:<30s}  {acc_pct:7.1f}%  {n:6d}  {correct:7d}")
        w(f"  {'OVERALL':<30s}  {overall_acc:7.1f}%  {total:6d}  {cases_df['_correct'].sum():7d}")
        w()

    # ── Confidence distribution ──
    w("CONFIDENCE SCORE DISTRIBUTION")
    w(SEP2)
    for label, n in conf_buckets.items():
        pct = n / total * 100
        bar = "█" * int(pct / 2)
        w(f"  {label:<10s}  {n:5d}  ({pct:5.1f}%)  {bar}")
    w()

    # ── Low confidence cases ──
    w("LOW-CONFIDENCE CASES — MANUAL REVIEW RECOMMENDED")
    w("-" * W)
    if len(low_conf) == 0:
        w("  None — all cases classified above 50% confidence.")
    else:
        w(f"  {'OPNAME':<40s}  {'PREDICTED CATEGORY':<25s}  {'CONF%':>5s}")
        w("  " + "-" * (W - 2))
        for _, row in low_conf.sort_values("ml_confidence_pct").iterrows():
            opname = str(row["opname"])[:38]
            cat    = str(row["ml_base_category"])[:23]
            conf   = row["ml_confidence_pct"]
            w(f"  {opname:<40s}  {cat:<25s}  {conf:5.1f}%")
    w()
    w(SEP)
    w("  END OF REPORT")
    w(SEP)

print(f"      Saved: {OUT_REPORT}")
print()
print(f"  Mean confidence : {mean_conf:.1f}%")
if has_optype:
    print(f"  Accuracy        : {overall_acc:.1f}%")
print(f"  Low conf cases  : {len(low_conf)}")
print("\nDone.")
