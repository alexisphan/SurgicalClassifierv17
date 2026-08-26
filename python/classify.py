"""
ClinicalBERT Surgical Case Classifier
--------------------------------------
Training data (combined):
  1. Full VitalDB case export  — 6,388 cases (opname + optype, 11 categories)
  2. Department neurosurgery reference — 133 entries (24 neuro categories)

Set VITALDB_CSV to the path of your full VitalDB export file.

Writes:
  ../output/bert_classified.csv
  ../output/bert_classification_report.txt

Run from the python/ folder:
    python classify.py
"""

import os
from datetime import datetime
from collections import Counter

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

# ── Set this to your full VitalDB export file path ─────────────────────────
VITALDB_CSV = os.path.join(DATA_DIR, "vitaldb_full_cases.csv")
# If the file is named differently, change the line above to match, e.g.:
# VITALDB_CSV = r"C:\Users\wishs\Downloads\caseid_subjectid_casestart_caseend_.txt"

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
NUM_EPOCHS = 40
BATCH_SIZE = 16
LR         = 2e-5
MAX_LEN    = 96   # longer to fit opname + dx together

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


# ── Build combined training set ────────────────────────────────────────────
print("[1/4] Building training set...")

train_texts  = []
train_labels_raw = []

# Layer 1: Full VitalDB case export (opname + dx → optype)
vdb = pd.DataFrame()   # default empty so report section always has it
if os.path.exists(VITALDB_CSV):
    vdb = pd.read_csv(VITALDB_CSV)
    vdb.columns = [c.strip() for c in vdb.columns]
    vdb = vdb.dropna(subset=["opname", "optype"])
    for _, row in vdb.iterrows():
        text = str(row["opname"]).strip()
        if "dx" in vdb.columns and pd.notna(row.get("dx")):
            text = text + " [SEP] " + str(row["dx"]).strip()
        train_texts.append(text)
        train_labels_raw.append(str(row["optype"]).strip())
    print(f"      VitalDB cases       : {len(vdb):,} rows")
else:
    print(f"      WARNING: VitalDB file not found at {VITALDB_CSV}")
    print(f"      Place your full VitalDB export in data/ as vitaldb_full_cases.csv")
    print(f"      Falling back to procedures_reference.csv only.")

# Layer 2: Department neurosurgery reference (neuro categories not in VitalDB)
dept = pd.read_csv(PROCEDURES_CSV)
dept.columns = [c.strip().strip('"') for c in dept.columns]
name_col = [c for c in dept.columns if "name" in c.lower()][0]
cat_col  = [c for c in dept.columns if "cat"  in c.lower()][0]

vdb_cats = set(train_labels_raw)
neuro_rows = dept[~dept[cat_col].str.strip().isin(vdb_cats)]

for _, row in neuro_rows.iterrows():
    train_texts.append(str(row[name_col]).strip())
    train_labels_raw.append(str(row[cat_col]).strip())

print(f"      Neurosurgery entries : {len(neuro_rows):,} rows")
print(f"      Total training set   : {len(train_texts):,} entries")

# Build label maps
labels_list = sorted(set(train_labels_raw))
label2id    = {l: i for i, l in enumerate(labels_list)}
id2label    = {i: l for l, i in label2id.items()}
train_labels = [label2id[l] for l in train_labels_raw]

print(f"      Total categories     : {len(labels_list)}")
print()
print("      Category distribution:")
for cat, n in sorted(Counter(train_labels_raw).items(), key=lambda x: -x[1]):
    print(f"        {cat:<35s}  {n:5d}")

# ── Tokenize & train ────────────────────────────────────────────────────────
print("\n[2/4] Loading tokenizer and fine-tuning ClinicalBERT...")
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
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
)

print(f"      Epochs: {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
trainer.train()

# ── Classify test cases ─────────────────────────────────────────────────────
print("\n[3/4] Classifying test cases...")
cases_df = pd.read_csv(CASES_CSV)
cases_df.columns = [c.strip().strip('"') for c in cases_df.columns]

# Use opname + dx for classification (same format as training)
if "dx" in cases_df.columns:
    texts = (cases_df["opname"].str.strip() + " [SEP] " +
             cases_df["dx"].fillna("").str.strip()).tolist()
else:
    texts = cases_df["opname"].str.strip().tolist()

# Process in batches to avoid memory issues with 2,290 cases
INFER_BATCH = 64
all_cats   = []
all_confs  = []

model.eval()
for i in range(0, len(texts), INFER_BATCH):
    batch_texts = texts[i : i + INFER_BATCH]
    enc = tokenizer(
        batch_texts, truncation=True, padding=True,
        max_length=MAX_LEN, return_tensors="pt"
    )
    with torch.no_grad():
        out = model(**enc)
    probs = torch.softmax(out.logits, dim=1)
    confs, pids = probs.max(dim=1)
    all_cats.extend([id2label[p.item()] for p in pids])
    all_confs.extend((confs.numpy() * 100).round(1).tolist())

cases_df["ml_base_category"] = all_cats
cases_df["ml_confidence_pct"] = all_confs
cases_df.to_csv(OUT_CSV, index=False)
print(f"      Saved: {OUT_CSV}")

# ── Generate report ─────────────────────────────────────────────────────────
print("\n[4/4] Generating classification report...")

total     = len(cases_df)
mean_conf = cases_df["ml_confidence_pct"].mean()
low_conf  = cases_df[cases_df["ml_confidence_pct"] < 50]

has_optype = "optype" in cases_df.columns
if has_optype:
    cases_df["_correct"] = (
        cases_df["ml_base_category"].str.strip() == cases_df["optype"].str.strip()
    )
    overall_acc = cases_df["_correct"].mean() * 100

pred_dist = Counter(cases_df["ml_base_category"].str.strip())

cat_acc = {}
if has_optype:
    for cat, grp in cases_df.groupby("optype"):
        acc = (grp["ml_base_category"].str.strip() == grp["optype"].str.strip()).mean()
        cat_acc[cat] = (acc * 100, len(grp))

conf_buckets = {}
for lo, hi, lbl in [
    (0,20,"<20%"),(20,30,"20-30%"),(30,40,"30-40%"),(40,50,"40-50%"),
    (50,60,"50-60%"),(60,70,"60-70%"),(70,80,"70-80%"),(80,90,"80-90%"),
    (90,101,"90-100%"),
]:
    n = ((cases_df["ml_confidence_pct"] >= lo) &
         (cases_df["ml_confidence_pct"] < hi)).sum()
    conf_buckets[lbl] = n

W   = 80
SEP = "=" * W
S2  = "-" * 50

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    def w(line=""): f.write(line + "\n")

    w(SEP)
    w("  ClinicalBERT SURGICAL CASE CLASSIFIER — CLASSIFICATION REPORT")
    w(f"  Model    : {MODEL_NAME}")
    w(f"  Epochs   : {NUM_EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LR}")
    w(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(SEP); w()

    w("SUMMARY"); w(S2)
    w(f"  Total cases classified      : {total:,}")
    w(f"  Training entries (VitalDB)  : {len(vdb):,}  case-level rows")
    w(f"  Training entries (neuro)    : {len(neuro_rows):,}  department reference")
    w(f"  Total training entries      : {len(train_texts):,}  ({len(labels_list)} categories)")
    w(f"  Average ML confidence       : {mean_conf:.1f}%")
    w(f"  Low confidence (<50%)       : {len(low_conf)}  ← review recommended")
    if has_optype:
        w(f"  Accuracy vs VitalDB optype  : {overall_acc:.1f}%"
          f"  ({cases_df['_correct'].sum():,}/{total:,} correct)")
    w()

    w("PREDICTED CATEGORY DISTRIBUTION"); w(S2)
    for cat, n in sorted(pred_dist.items(), key=lambda x: -x[1]):
        w(f"  {cat:<35s}  {n:5d}  ({n/total*100:5.1f}%)")
    w()

    if has_optype:
        w("ACCURACY VS VITALDB GROUND TRUTH (optype)"); w(S2)
        w(f"  {'Category':<30s}  {'Accuracy':>8s}  {'Cases':>6s}  {'Correct':>7s}")
        w("  " + "-" * 58)
        for cat in sorted(cat_acc):
            acc_pct, n = cat_acc[cat]
            w(f"  {cat:<30s}  {acc_pct:7.1f}%  {n:6d}  {round(acc_pct/100*n):7d}")
        w(f"  {'OVERALL':<30s}  {overall_acc:7.1f}%  {total:6d}"
          f"  {cases_df['_correct'].sum():7d}")
        w()

    w("CONFIDENCE SCORE DISTRIBUTION"); w(S2)
    for lbl, n in conf_buckets.items():
        pct = n / total * 100
        w(f"  {lbl:<10s}  {n:5d}  ({pct:5.1f}%)  {'█' * int(pct/2)}")
    w()

    w("LOW-CONFIDENCE CASES — MANUAL REVIEW RECOMMENDED")
    w("-" * W)
    if len(low_conf) == 0:
        w("  None — all cases classified above 50% confidence.")
    else:
        w(f"  {'OPNAME':<40s}  {'PREDICTED CATEGORY':<25s}  {'CONF%':>5s}")
        w("  " + "-" * (W - 2))
        for _, row in low_conf.sort_values("ml_confidence_pct").iterrows():
            w(f"  {str(row['opname'])[:38]:<40s}"
              f"  {str(row['ml_base_category'])[:23]:<25s}"
              f"  {row['ml_confidence_pct']:5.1f}%")
    w(); w(SEP); w("  END OF REPORT"); w(SEP)

print(f"      Saved: {OUT_REPORT}")
print(f"\n  Mean confidence : {mean_conf:.1f}%")
if has_optype:
    print(f"  Accuracy        : {overall_acc:.1f}%")
print(f"  Low conf cases  : {len(low_conf)}")
print("\nDone.")
