#!/usr/bin/env python3
# fermi_finish_from_xlsx.py
"""
Consume annotated responses.xlsx and produce histograms + summary metrics using the HUMAN labels.

What it does
------------
- Reads the "responses" sheet of an XLSX produced by fermi_collect_to_xlsx.py (after annotation).
- Uses the k_human column to build histograms per (question, model_kind) and draws the true_k line.
- Writes:
    * plots_annotated/qXXXX_model.png
    * summary.csv (per question/model row with majority-vote accuracy and counts)
    * overall_metrics.txt

Usage
-----
python fermi_finish_from_xlsx.py --xlsx responses100-200.xlsx --out_dir results_sc_annotated
"""
import argparse
import os
from collections import Counter
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt

def plot_hist(exponents: List[int], true_k: Optional[int], title: str, out_path: str):
    if not exponents:
        plt.figure()
        plt.title(title + " (no labeled samples)")
        plt.xlabel("Exponent k (human)")
        plt.ylabel("Count")
        if true_k is not None:
            plt.axvline(x=int(true_k), linestyle="--", linewidth=2, label=f"True k = {int(true_k)}")
            plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    xs = sorted(set(exponents + ([int(true_k)] if true_k is not None else [])))
    full_range = list(range(min(xs), max(xs) + 1))
    counts = Counter(exponents)
    ys = [counts.get(x, 0) for x in full_range]

    plt.figure()
    plt.bar(full_range, ys)
    if true_k is not None:
        plt.axvline(x=int(true_k), linestyle="--", linewidth=2, label=f"True k = {int(true_k)}")
        plt.legend()
    for x, y in zip(full_range, ys):
        if y > 0:
            plt.text(x, y, str(y), ha="center", va="bottom")
    plt.title(title)
    plt.xlabel("Exponent k (human)")
    plt.ylabel(f"Count (N={len(exponents)})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Finish pipeline from annotated XLSX.")
    ap.add_argument("--xlsx", type=str, required=True, help="Path to responses.xlsx with k_human filled in.")
    ap.add_argument("--out_dir", type=str, default="results_sc_annotated")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots_annotated")
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_excel(args.xlsx, sheet_name="responses")

    # Keep only rows that have k_human
    df_anno = df[pd.notna(df["k_human"])].copy()
    # Coerce to int where possible
    df_anno["k_human"] = df_anno["k_human"].astype(int)

    summary_rows = []
    grouped = df_anno.groupby(["q_index", "model_kind"], as_index=False)
    for (q_idx, kind), g in grouped:
        true_k = g["true_k"].iloc[0] if pd.notna(g["true_k"]).any() else None
        if pd.notna(true_k):
            try:
                true_k = int(true_k)
            except Exception:
                true_k = None

        exps = g["k_human"].tolist()
        # Majority vote
        vote_k, vote_count = None, 0
        if exps:
            c = Counter(exps)
            vote_k, vote_count = c.most_common(1)[0]

        acc_majority = (vote_k == true_k) if (vote_k is not None and true_k is not None) else None

        title = f"Q{int(q_idx)+1} [{kind}] — Human-annotated"
        if true_k is not None:
            title += f" | True k={true_k}"
        out_path = os.path.join(plots_dir, f"q{int(q_idx):04d}_{kind}.png")
        plot_hist(exps, true_k, title, out_path)

        summary_rows.append({
            "q_index": int(q_idx),
            "model_kind": kind,
            "n_labeled": len(exps),
            "majority_k": vote_k,
            "majority_count": vote_count,
            "true_k": true_k,
            "accuracy_majority": acc_majority,
            "plot_path": out_path
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["q_index", "model_kind"])
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Overall metrics by model
    overall = summary_df.groupby("model_kind")["accuracy_majority"].mean().reset_index()
    metrics_path = os.path.join(args.out_dir, "overall_metrics.txt")
    with open(metrics_path, "w") as f:
        for _, row in overall.iterrows():
            f.write(f"{row['model_kind']}: majority accuracy = {row['accuracy_majority']:.3f}\n")

    print(f"Wrote: {summary_csv}")
    print(f"Wrote plots under: {plots_dir}")
    print(f"Wrote: {metrics_path}")

if __name__ == "__main__":
    main()