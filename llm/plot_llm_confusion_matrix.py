#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a normalized confusion matrix for LLM predictions.

Default input:
  data/test/test_predictions.csv

Default output:
  data/test/llm_confusion_matrix_normalized.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

DEFAULT_INPUT = ROOT / "data" / "test" / "test_predictions.csv"
DEFAULT_OUTPUT = ROOT / "data" / "test" / "llm_confusion_matrix_normalized.png"
LABELS = ["very_negative", "negative", "neutral", "positive", "very_positive"]


def normalize_label(value: object) -> str:
    return str(value).strip().lower()


def plot_confusion_matrix(args: argparse.Namespace) -> int:
    input_csv = Path(args.input_csv).resolve()
    output_png = Path(args.output_png).resolve()

    df = pd.read_csv(input_csv, encoding=args.encoding)
    missing = [col for col in (args.true_col, args.pred_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột {missing}. Cột hiện có: {list(df.columns)}")

    y_true = df[args.true_col].map(normalize_label)
    y_pred = df[args.pred_col].map(normalize_label)

    valid = y_true.isin(LABELS) & y_pred.isin(LABELS)
    dropped = int((~valid).sum())
    if not valid.any():
        raise ValueError("Không có dòng hợp lệ để vẽ confusion matrix.")

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    normalize = "true" if args.normalized else None
    values_format = ".2f" if args.normalized else "d"
    title = args.title or ("Confusion Matrix (Normalized)" if args.normalized else "Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize=normalize)
    accuracy = float((y_true == y_pred).mean())

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=args.dpi)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(cmap="Blues", values_format=values_format, ax=ax, colorbar=True)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)

    print(f"Saved: {output_png}")
    print(f"Rows used: {len(y_true)}")
    print(f"Rows dropped: {dropped}")
    print(f"Accuracy: {accuracy:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot LLM confusion matrix from prediction CSV.")
    parser.add_argument("--input_csv", default=str(DEFAULT_INPUT), help="CSV có cột label và llm_pred.")
    parser.add_argument("--output_png", default=str(DEFAULT_OUTPUT), help="Đường dẫn PNG output.")
    parser.add_argument("--true_col", default="label")
    parser.add_argument("--pred_col", default="llm_pred")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--title", default="")
    parser.add_argument("--normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show", action="store_true", help="Hiển thị cửa sổ matplotlib sau khi lưu.")
    return parser


if __name__ == "__main__":
    raise SystemExit(plot_confusion_matrix(build_parser().parse_args()))
