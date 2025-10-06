# scripts/nlp_analyze.py
from __future__ import annotations
import json, math, os, re, sys, glob
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# NLTK sentiment + stopwords (download at runtime if missing)
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
try:
    from nltk.corpus import stopwords
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

# Ensure VADER is present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

SITE = Path("site"); SITE.mkdir(parents=True, exist_ok=True)

# Helpers
URL_RE = re.compile(r"https?://\S+")
TOKEN_RE = re.compile(r"[#@]?[a-z0-9']{2,}")

STOP = set(stopwords.words("english"))
STOP |= {"rt", "amp", "https", "http", "t", "co", "www", "com"}

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = URL_RE.sub(" ", text)
    toks = TOKEN_RE.findall(text)
    out = []
    for t in toks:
        if t.startswith("@"):
            continue  # ignore mentions
        if t.startswith("#"):
            t = t[1:]
        if not t or t in STOP or t.isdigit():
            continue
        out.append(t.strip("'"))
    return out

def ngrams(tokens, n=2):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def newest_parquet():
    files = sorted(glob.glob("data/processed/*.parquet"))
    if not files:
        raise SystemExit("[error] no parquet files found in data/processed/")
    return files[-1]

def lift_terms(recent_counts: Counter, base_counts: Counter, k_min_recent=3, topk=20):
    vocab = set(recent_counts) | set(base_counts)
    V = max(1, len(vocab))
    recent_total = sum(recent_counts.values())
    base_total = max(1, sum(base_counts.values()))
    scored = []
    for term, r in recent_counts.items():
        if r < k_min_recent:
            continue
        b = base_counts.get(term, 0)
        # add-1 smoothing with vocabulary in denominator; compute "lift"
        recent_rate = (r + 1) / (recent_total + V)
        base_rate   = (b + 1) / (base_total + V)
        score = recent_rate / base_rate
        scored.append((term, r, b, score))
    # sort by lift, then recent count
    scored.sort(key=lambda x: (x[3], x[1]), reverse=True)
    return scored[:topk]

def evergreen_terms(recent_counts: Counter, base_counts: Counter, tol=0.15, k_min=10, topk=20):
    # Terms that are consistently present (high counts) with lift ~1
    combined = []
    recent_total = sum(recent_counts.values())
    base_total = max(1, sum(base_counts.values()))
    vocab = set(recent_counts) | set(base_counts)
    V = max(1, len(vocab))
    for term in vocab:
        r = recent_counts.get(term, 0)
        b = base_counts.get(term, 0)
        if r + b < k_min:
            continue
        recent_rate = (r + 1) / (recent_total + V)
        base_rate   = (b + 1) / (base_total + V)
        L = recent_rate / base_rate
        if abs(L - 1.0) <= tol:
            combined.append((term, r + b, L))
    combined.sort(key=lambda x: (x[1], -abs(x[2] - 1.0)), reverse=True)
    return combined[:topk]

def main():
    path = newest_parquet()
    df = pd.read_parquet(path)

    # Ensure datetimes
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    else:
        df["created_at"] = pd.NaT

    # Sentiment (VADER) over the most recent window (7 days if present; else last 500 posts)
    sia = SentimentIntensityAnalyzer()

    now_utc = pd.Timestamp.utcnow()
    cutoff_recent = now_utc - pd.Timedelta(days=7)
    if df["created_at"].n_]()
