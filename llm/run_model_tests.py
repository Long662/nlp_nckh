#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch test PhoBERT and Ollama LLM on a CSV file.

Default input:
  data/test/test.csv

Default output:
  data/test/test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests
from requests.exceptions import ReadTimeout


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
PHOBERT_DIR = ROOT / "phobert"
DEFAULT_INPUT = ROOT / "data" / "test" / "test.csv"
DEFAULT_OUTPUT = ROOT / "data" / "test" / "test_predictions.csv"
DEFAULT_SUMMARY = ROOT / "data" / "test" / "test_predictions_summary.json"
OLLAMA_URL = "http://localhost:11434/api/generate"

LABELS = ["very_negative", "negative", "neutral", "positive", "very_positive"]
LABELS_BY_LENGTH = sorted(LABELS, key=len, reverse=True)


SYSTEM_PROMPT = """Bạn là bộ phân loại cảm xúc bình luận sản phẩm tiếng Việt.
Nhãn hợp lệ: very_positive, positive, neutral, negative, very_negative.

Nhiệm vụ:
- Đọc bình luận sản phẩm.
- Chọn MỘT nhãn duy nhất thể hiện cảm xúc tổng thể.
- CHỈ trả về đúng MỘT từ trong các từ sau, không giải thích:
very_positive
positive
neutral
negative
very_negative
"""


def normalize_label(value: object) -> str:
    return str(value).strip().lower()


def extract_label(raw: object) -> str:
    if raw is None:
        return "neutral"

    s = unicodedata.normalize("NFKC", str(raw)).strip().lower()
    if not s:
        return "neutral"

    for label in LABELS_BY_LENGTH:
        if re.search(rf"(?<![a-z_]){re.escape(label)}(?![a-z_])", s):
            return label

    alias_map = {
        "very positive": "very_positive",
        "verypositive": "very_positive",
        "rat tot": "very_positive",
        "rất tốt": "very_positive",
        "tuyệt vời": "very_positive",
        "positive": "positive",
        "tot": "positive",
        "tốt": "positive",
        "good": "positive",
        "neutral": "neutral",
        "trung tinh": "neutral",
        "trung tính": "neutral",
        "binh thuong": "neutral",
        "bình thường": "neutral",
        "on": "neutral",
        "ổn": "neutral",
        "negative": "negative",
        "xau": "negative",
        "xấu": "negative",
        "te": "negative",
        "tệ": "negative",
        "bad": "negative",
        "very negative": "very_negative",
        "verynegative": "very_negative",
        "rat te": "very_negative",
        "rất tệ": "very_negative",
        "kinh khung": "very_negative",
        "kinh khủng": "very_negative",
        "toi te": "very_negative",
        "tồi tệ": "very_negative",
    }
    for key, label in alias_map.items():
        if key in s:
            return label

    tokens = re.findall(r"[a-z_]+", s)
    for token in tokens:
        token = token.replace("-", "_").strip("._ ")
        if token in LABELS:
            return token
        if token in alias_map:
            return alias_map[token]

    return "neutral"


def parse_phobert_output(output: str) -> str:
    match = re.search(r"Pred:\s*([A-Za-z_]+)", output)
    return extract_label(match.group(1) if match else output)


def load_phobert_runner(module_name: str) -> Callable[[str], str]:
    if str(PHOBERT_DIR) not in sys.path:
        sys.path.insert(0, str(PHOBERT_DIR))

    module = __import__(module_name)
    infer = getattr(module, "infer", None)
    if not callable(infer):
        raise AttributeError(f"Module {module_name!r} không có hàm infer(text)")
    return infer


def call_ollama(text: str, model: str, url: str, timeout: int) -> tuple[str, str]:
    prompt = f'{SYSTEM_PROMPT}\n\nBình luận: "{text}"\nNhãn:'
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "num_ctx": 512,
            "num_predict": 12,
            "stop": ["\n"],
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=(10, timeout))
    except ReadTimeout as exc:
        raise RuntimeError(
            f"Ollama timeout sau {timeout}s. Model {model} có thể quá nặng hoặc đang chạy quá chậm."
        ) from exc

    if not response.ok:
        detail = http_error_detail(response)
        raise RuntimeError(f"Ollama HTTP {response.status_code}: {detail or response.reason}")

    raw = response.json().get("response", "").strip()
    return extract_label(raw), raw


def http_error_detail(response: requests.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return response.text.strip()


def accuracy(y_true: pd.Series, y_pred: pd.Series) -> Optional[float]:
    mask = y_true.notna() & y_pred.notna() & (y_pred.astype(str).str.len() > 0)
    if not mask.any():
        return None
    lhs = y_true[mask].map(normalize_label)
    rhs = y_pred[mask].map(normalize_label)
    return float((lhs == rhs).mean())


def mean_seconds(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def fill_correct_columns(df: pd.DataFrame, has_label: bool, do_phobert: bool, do_llm: bool) -> None:
    if not has_label:
        return

    labels = df["label"].map(normalize_label)
    if do_phobert:
        preds = df["phobert_pred"].map(normalize_label)
        df["phobert_correct"] = (labels == preds) & preds.isin(LABELS)
    if do_llm:
        preds = df["llm_pred"].map(normalize_label)
        df["llm_correct"] = (labels == preds) & preds.isin(LABELS)


def run(args: argparse.Namespace) -> int:
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.out_csv).resolve()
    summary_json = Path(args.summary_json).resolve() if args.summary_json else None

    df = pd.read_csv(input_csv, encoding=args.encoding)
    if "text" not in df.columns:
        raise ValueError(f"CSV thiếu cột 'text'. Cột hiện có: {list(df.columns)}")

    empty_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if empty_cols and args.drop_unnamed:
        df = df.drop(columns=empty_cols)

    if args.limit:
        df = df.head(args.limit).copy()

    total = len(df)
    has_label = "label" in df.columns
    do_phobert = args.models in ("both", "phobert")
    do_llm = args.models in ("both", "llm")

    phobert_infer: Optional[Callable[[str], str]] = None
    if do_phobert:
        print(f"[PhoBERT] Loading module {args.phobert_module} ...", flush=True)
        phobert_infer = load_phobert_runner(args.phobert_module)

    for col in [
        "phobert_pred",
        "phobert_correct",
        "phobert_infer_time_s",
        "phobert_error",
        "llm_pred",
        "llm_correct",
        "llm_raw",
        "llm_infer_time_s",
        "llm_error",
    ]:
        if col not in df.columns:
            df[col] = ""

    started_all = time.perf_counter()

    for idx, row in df.iterrows():
        text = "" if pd.isna(row["text"]) else str(row["text"])
        row_no = int(idx) + 1

        if do_phobert and phobert_infer is not None:
            started = time.perf_counter()
            try:
                raw = phobert_infer(text)
                df.at[idx, "phobert_pred"] = parse_phobert_output(raw)
                df.at[idx, "phobert_error"] = ""
            except Exception as exc:
                df.at[idx, "phobert_pred"] = ""
                df.at[idx, "phobert_error"] = str(exc)
            finally:
                df.at[idx, "phobert_infer_time_s"] = round(time.perf_counter() - started, 6)

        if do_llm:
            started = time.perf_counter()
            try:
                label, raw = call_ollama(text, args.llm_model, args.ollama_url, args.ollama_timeout)
                df.at[idx, "llm_pred"] = label
                df.at[idx, "llm_raw"] = raw
                df.at[idx, "llm_error"] = ""
            except Exception as exc:
                df.at[idx, "llm_pred"] = ""
                df.at[idx, "llm_raw"] = ""
                df.at[idx, "llm_error"] = str(exc)
            finally:
                df.at[idx, "llm_infer_time_s"] = round(time.perf_counter() - started, 6)

        if row_no % args.progress_every == 0 or row_no == total:
            print(f"Processed {row_no}/{total}", flush=True)

        if args.save_every and row_no % args.save_every == 0:
            fill_correct_columns(df, has_label, do_phobert, do_llm)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    elapsed = time.perf_counter() - started_all
    fill_correct_columns(df, has_label, do_phobert, do_llm)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows": total,
        "elapsed_time_s": round(elapsed, 6),
        "models": args.models,
        "llm_model": args.llm_model if do_llm else None,
        "phobert_module": args.phobert_module if do_phobert else None,
        "has_label": has_label,
        "phobert_accuracy": accuracy(df["label"], df["phobert_pred"]) if has_label and do_phobert else None,
        "llm_accuracy": accuracy(df["label"], df["llm_pred"]) if has_label and do_llm else None,
        "phobert_mean_infer_time_s": mean_seconds(df["phobert_infer_time_s"]) if do_phobert else None,
        "llm_mean_infer_time_s": mean_seconds(df["llm_infer_time_s"]) if do_llm else None,
        "phobert_error_count": int((df["phobert_error"].astype(str) != "").sum()) if do_phobert else None,
        "llm_error_count": int((df["llm_error"].astype(str) != "").sum()) if do_llm else None,
    }

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch test PhoBERT and Ollama LLM on a CSV dataset.")
    parser.add_argument("--input_csv", default=str(DEFAULT_INPUT), help="CSV input có cột text và tùy chọn label.")
    parser.add_argument("--out_csv", default=str(DEFAULT_OUTPUT), help="CSV output có thêm prediction/time/error columns.")
    parser.add_argument("--summary_json", default=str(DEFAULT_SUMMARY), help="File JSON summary. Để rỗng nếu không muốn ghi.")
    parser.add_argument("--encoding", default="utf-8", help="Encoding đọc CSV input.")
    parser.add_argument("--models", choices=["both", "phobert", "llm"], default="both")
    parser.add_argument("--phobert_module", default="my_phobert_only")
    parser.add_argument("--llm_model", default="qwen3:8b")
    parser.add_argument("--ollama_url", default=OLLAMA_URL)
    parser.add_argument("--ollama_timeout", type=int, default=240)
    parser.add_argument("--limit", type=int, default=0, help="Chạy thử N dòng đầu. 0 nghĩa là toàn bộ file.")
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=25, help="Checkpoint CSV mỗi N dòng. 0 để tắt.")
    parser.add_argument(
        "--drop_unnamed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bỏ các cột Unnamed do dấu phẩy thừa.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
