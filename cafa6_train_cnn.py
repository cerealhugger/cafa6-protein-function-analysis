#!/usr/bin/env python3

"""
CAFA6 Protein Function Prediction - CNN Head over frozen ESM2
Usage:
  # Original usage with absolute paths (commented out):
  # python cafa6_train_cnn.py \
  #   --base_path "/Users/zhouheng/Desktop/kaggle/cafa-6-protein-function-prediction" \
  #   --epochs 5 --batch_size 8 --top_k 50 \
  #   --submission_path "/Users/zhouheng/Desktop/kaggle/submission_cnn.tsv"
  
  # New usage with relative paths:
  python cafa6_train_cnn.py \
    --base_path "." \
    --epochs 5 --batch_size 8 --top_k 50 \
    --submission_path "./submission_cnn.tsv"
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from tqdm import tqdm


# Helpers
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple Metal
    return "cpu"

def fail(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)

def parse_args():
    ap = argparse.ArgumentParser(description="CAFA6 - CNN over frozen ESM2 baseline")
    # ap.add_argument("--base_path", type=str,
    #                 default="/Users/zhouheng/Desktop/kaggle/cafa-6-protein-function-prediction",
    #                 help="Folder containing Train/, Test/, IA.tsv, etc.")
    ap.add_argument("--base_path", type=str,
                    default=".",
                    help="Folder containing Train/, Test/, IA.tsv, etc.")
    # ap.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D",
    #                 help="HuggingFace model id for ESM2")
    
    ap.add_argument("--model_name", type=str, default="facebook/esm2_t48_15B_UR50D",
                    help="HuggingFace model id for ESM2")

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_length", type=int, default=1022)
    ap.add_argument("--top_k", type=int, default=50, help="Use top-K frequent GO terms")
    ap.add_argument("--threshold", type=float, default=0.5, help="prob→label threshold")
    # ap.add_argument("--submission_path", type=str, default="submission_cnn.tsv")
    ap.add_argument("--submission_path", type=str, default="./submission_cnn.tsv")
    ap.add_argument("--predict_split", type=str, choices=["trainval", "test"], default="trainval",
                    help="Generate submission for train+val proteins or test superset")
    ap.add_argument("--sample_n", type=int, default=0,
                    help="(Optional) limit number of training proteins for quick smoke test")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# IO & Data
def robust_glob(root: Path, pattern: str) -> Path:
    hits = list(root.rglob(pattern))
    return hits[0] if hits else None

def load_tables(base: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    train_path = None
    for cand in ["Train", "train", "Training", "training"]:
        p = base / cand
        if p.exists():
            train_path = p
            break
    if train_path is None:
        hit = robust_glob(base, "train_terms.tsv")
        if hit: train_path = hit.parent
    if train_path is None:
        fail("Could not locate Train folder or train_terms.tsv")

    test_path = None
    for cand in ["Test", "test", "PublicTest", "testing"]:
        p = base / cand
        if p.exists():
            test_path = p
            break
    if test_path is None:
        hit = robust_glob(base, "test*superset*.fasta*")
        if hit: test_path = hit.parent

    tt = pd.read_csv(train_path/"train_terms.tsv", sep="\t",
                     header=None, names=["protein_id", "go_term", "ontology"])
    tax = pd.read_csv(train_path/"train_taxonomy.tsv", sep="\t",
                      header=None, names=["protein_id", "taxon_id"])

    train_fasta = train_path/"train_sequences.fasta"
    test_fasta = None
    if test_path:
        cand = test_path/"testsuperset.fasta"
        if cand.exists():
            test_fasta = cand
        else:
            hit = robust_glob(test_path, "test*superset*.fasta*")
            test_fasta = hit if hit else None

    return tt, tax, train_fasta, test_fasta

def parse_protein_id(record_id: str) -> str:
    rid = record_id.split()[0]
    if "|" in rid:
        parts = rid.split("|")
        return parts[1] if len(parts) > 1 and parts[1] else parts[0]
    return rid

def load_fasta_dict(fasta_path: Path) -> Dict[str, str]:
    try:
        from Bio import SeqIO
    except Exception:
        fail("Missing dependency: biopython. Please `pip install biopython`.")
    seqs = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        pid = parse_protein_id(str(rec.id))
        seqs[pid] = str(rec.seq)
    return seqs

# Datasets
class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, pid_list: List[str], y_tensor: torch.Tensor, seq_dict: Dict[str, str]):
        self.pids = pid_list
        self.y = y_tensor
        self.seq = seq_dict
    def __len__(self):
        return len(self.pids)
    def __getitem__(self, i):
        pid = self.pids[i]
        return pid, self.seq[pid], self.y[i]

# Model: CNN head
class CNNHead(nn.Module):
    def __init__(self, in_dim: int, num_labels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 256, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, num_labels)
    def forward(self, token_feats: torch.Tensor) -> torch.Tensor:
        x = token_feats.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        return logits

# Collate
def make_collate(tokenizer, max_length: int):
    def collate(batch):
        pids, seqs, ys = zip(*batch)
        tokens = tokenizer(
            list(seqs),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        ys = torch.stack(ys, dim=0)
        return list(pids), tokens, ys
    return collate

# Train / Eval (with tqdm)
def run_epoch(backbone, head, loader, device, criterion, optimizer=None, threshold=0.5):
    train_mode = optimizer is not None
    head.train(train_mode)
    if train_mode: backbone.eval()
    total_loss = 0.0
    preds_all, trues_all = [], []

    for pids, tokens, ys in tqdm(loader, desc="train" if train_mode else "eval", total=len(loader)):
        ys = ys.to(device)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            feats = backbone(**tokens).last_hidden_state

        logits = head(feats)
        loss = criterion(logits, ys)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * ys.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= threshold).astype(np.int32)
        preds_all.append(preds)
        trues_all.append(ys.detach().cpu().numpy())

    preds_all = np.vstack(preds_all)
    trues_all = np.vstack(trues_all)
    f1 = f1_score(trues_all, preds_all, average="micro", zero_division=0)
    return total_loss / len(loader.dataset), f1

# Submission builder
def build_submission(head, backbone, tokenizer, seq_dict: Dict[str, str], pids: List[str],
                     mlb: MultiLabelBinarizer, device, max_length: int,
                     out_path: Path, keep_top_per_protein: int = 1500, min_score: float = 0.01):
    head.eval(); backbone.eval()
    ds = ProteinDataset(pids, torch.zeros((len(pids), len(mlb.classes_)), dtype=torch.float32), seq_dict)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=make_collate(tokenizer, max_length))

    all_rows = []
    with torch.no_grad():
        for pids_b, tokens, _ in tqdm(loader, desc="predict", total=len(loader)):
            tokens = {k: v.to(device) for k, v in tokens.items()}
            feats = backbone(**tokens).last_hidden_state
            logits = head(feats)
            probs = torch.sigmoid(logits).cpu().numpy()
            for pid, prob in zip(pids_b, probs):
                go_scores = [(go, float(s)) for go, s in zip(mlb.classes_, prob) if s > min_score]
                if not go_scores:
                    continue
                go_scores.sort(key=lambda x: x[1], reverse=True)
                go_scores = go_scores[:keep_top_per_protein]
                for go, s in go_scores:
                    s = max(min(s, 1.0), 1e-6)
                    all_rows.append((pid, go, s))

    sub = pd.DataFrame(all_rows, columns=["protein_id", "go_term", "score"])
    sub = sub.sort_values(["protein_id", "score"], ascending=[True, False]) \
             .groupby("protein_id").head(keep_top_per_protein)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, sep="\t", index=False, header=False)
    print(f"[OK] Submission written: {out_path}  (lines={len(sub)})")
    return sub

# Main
def main():
    args = parse_args()
    set_seed(args.seed)

    try:
        from transformers import AutoTokenizer, AutoModel
    except Exception:
        fail("Missing dependency: transformers. Please `pip install transformers`.")

    base = Path(args.base_path)
    if not base.exists():
        fail(f"Base path not found: {base}")

    print("[Info] Loading tables & FASTA paths ...")
    train_terms, train_taxonomy, train_fasta, test_fasta = load_tables(base)
    if not train_fasta or not Path(train_fasta).exists():
        fail("train_sequences.fasta not found.")

    print("[Info] Loading sequences ...")
    train_sequences = load_fasta_dict(train_fasta)
    test_sequences = load_fasta_dict(test_fasta) if test_fasta else {}

    print(f"[Info] Building label space (Top-K={args.top_k}) ...")
    subset = train_terms.copy()
    top_terms = subset["go_term"].value_counts().head(args.top_k).index
    subset = subset[subset["go_term"].isin(top_terms)]
    labels_df = subset.groupby("protein_id")["go_term"].apply(list).reset_index()
    labels_df = labels_df[labels_df["protein_id"].isin(train_sequences.keys())].reset_index(drop=True)

    if args.sample_n and args.sample_n > 0:
        labels_df = labels_df.sample(n=min(args.sample_n, len(labels_df)), random_state=args.seed).reset_index(drop=True)
        print(f"[Info] Using sample_n={len(labels_df)} proteins for a quick run.")

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels_df["go_term"].tolist()).astype(np.float32)
    pid_list = labels_df["protein_id"].tolist()

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(pid_list)); rng.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    pids_train = [pid_list[i] for i in train_idx]
    pids_val   = [pid_list[i] for i in val_idx]
    Y_train = torch.tensor(Y[train_idx], dtype=torch.float32)
    Y_val   = torch.tensor(Y[val_idx], dtype=torch.float32)

    print(f"[Info] Train/Val sizes: {len(pids_train)} / {len(pids_val)} ; labels={len(mlb.classes_)}")

    # Backbone (frozen)
    print(f"[Info] Loading backbone: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    backbone = AutoModel.from_pretrained(args.model_name)
    for p in backbone.parameters():
        p.requires_grad = False

    device = pick_device()
    print(f"[Info] Device: {device}")
    backbone.to(device).eval()
    hidden_dim = backbone.config.hidden_size

    # Head
    head = CNNHead(in_dim=hidden_dim, num_labels=len(mlb.classes_)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Dataloaders
    train_ds = ProteinDataset(pids_train, Y_train, train_sequences)
    val_ds   = ProteinDataset(pids_val,   Y_val,   train_sequences)
    collate  = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Train
    print("[Info] Starting training ... (showing tqdm per-batch ETA)")
    best_f1, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_f1 = run_epoch(backbone, head, train_loader, device, criterion, optimizer, args.threshold)
        va_loss, va_f1 = run_epoch(backbone, head, val_loader,   device, criterion, None,      args.threshold)
        print(f"[Epoch {ep}] train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} | val_loss={va_loss:.4f} val_f1={va_f1:.4f}")
        if va_f1 > best_f1:
            best_f1, best_state = va_f1, head.state_dict().copy()

    if best_state:
        head.load_state_dict(best_state)
    print(f"[Result] Best val micro-F1: {best_f1:.4f}")

    # Submission target split
    if args.predict_split == "trainval":
        target_pids = pids_train + pids_val
        seq_source = train_sequences
    else:
        if not test_sequences:
            fail("No test superset FASTA found. Use --predict_split trainval or provide Test/testsuperset.fasta.")
        target_pids = list(test_sequences.keys())
        seq_source = test_sequences

    out_path = Path(args.submission_path)
    print(f"[Info] Building submission for {len(target_pids)} proteins → {out_path}")
    build_submission(
        head=head,
        backbone=backbone,
        tokenizer=tokenizer,
        seq_dict=seq_source,
        pids=target_pids,
        mlb=mlb,
        device=device,
        max_length=args.max_length,
        out_path=out_path,
        keep_top_per_protein=1500,
        min_score=0.01
    )

if __name__ == "__main__":
    main()