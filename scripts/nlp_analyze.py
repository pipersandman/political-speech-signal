# scripts/nlp_analyze.py
from __future__ import annotations
import json, math, os, re, sys, glob
from collections import Counter
from datetime import datetime, timezone
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
            continue
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
    recent_total = sum(recent_counts.values()) or 1
    base_total = max(1, sum(base_counts.values()), 1)
    scored = []
    for term, r in recent_counts.items():
        if r < k_min_recent:
            continue
        b = base_counts.get(term, 0)
        recent_rate = (r + 1) / (recent_total + V)
        base_rate   = (b + 1) / (base_total + V)
        score = recent_rate / base_rate
        scored.append((term, r, b, score))
    scored.sort(key=lambda x: (x[3], x[1]), reverse=True)
    return scored[:topk]

def evergreen_terms(recent_counts: Counter, base_counts: Counter, tol=0.15, k_min=10, topk=20):
    combined = []
    recent_total = sum(recent_counts.values()) or 1
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

    # Ensure datetime column exists and is timezone-aware
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    else:
        df["created_at"] = pd.NaT

    sia = SentimentIntensityAnalyzer()

    now_utc = pd.Timestamp.utcnow()
    cutoff_recent = now_utc - pd.Timedelta(days=7)

    # Recent window (prefer last 7 days; fallback to newest 500 rows)
    if df["created_at"].notna().any():
        recent_mask = df["created_at"] >= cutoff_recent
        df_recent = df[recent_mask].copy()
        if len(df_recent) < 200:
            df_recent = df.sort_values("created_at", ascending=False).head(500).copy()
    else:
        df_recent = df.head(500).copy()

    # Baseline (previous ~180 days; fallback to older slice)
    if df["created_at"].notna().any():
        cutoff_base = cutoff_recent - pd.Timedelta(days=180)
        df_base = df[(df["created_at"] < cutoff_recent) & (df["created_at"] >= cutoff_base)].copy()
        if len(df_base) < 1000:
            df_base = df.sort_values("created_at", ascending=True).head(5000).copy()
    else:
        df_base = df.tail(5000).copy()

    # Sentiment scores on recent window
    recent_texts = df_recent["text"].fillna("").astype(str).tolist()
    rec_scores = [sia.polarity_scores(t)["compound"] for t in recent_texts] if recent_texts else []

    def tone_bucket(c):
        if c >= 0.6:  return "very_positive"
        if c >= 0.2:  return "positive"
        if c > -0.2:  return "neutral"
        if c > -0.6:  return "negative"
        return "very_negative"

    buckets = Counter(tone_bucket(c) for c in rec_scores)
    total_scored = sum(buckets.values()) or 1
    dist_pct = {k: round(100*b/total_scored, 1) for k, b in buckets.items()}
    mean_compound = round(sum(rec_scores)/len(rec_scores), 3) if rec_scores else 0.0

    # Token counts
    rec_tokens = [tokenize(t) for t in recent_texts]
    base_tokens = [tokenize(t) for t in df_base["text"].fillna("").astype(str).tolist()]

    rec_unigrams = Counter(t for toks in rec_tokens for t in toks)
    base_unigrams = Counter(t for toks in base_tokens for t in toks)

    rec_bigrams = Counter(bg for toks in rec_tokens for bg in ngrams(toks, 2))
    rec_trigrams = Counter(tg for toks in rec_tokens for tg in ngrams(toks, 3))

    rising = [
        {"term": t, "recent": r, "baseline": b, "lift": round(L, 2)}
        for (t, r, b, L) in lift_terms(rec_unigrams, base_unigrams, k_min_recent=3, topk=15)
    ]
    evergreen = [
        {"term": t, "combined": c, "lift": round(L, 2)}
        for (t, c, L) in evergreen_terms(rec_unigrams, base_unigrams, tol=0.15, k_min=20, topk=15)
    ]

    top_bi = [{"ngram": n, "count": c} for n, c in rec_bigrams.most_common(10) if c >= 3]
    top_tri = [{"ngram": n, "count": c} for n, c in rec_trigrams.most_common(10) if c >= 3]

    # Examples: positive / negative / neutral (guard against empty)
    df_recent = df_recent.copy()
    if len(df_recent) and len(rec_scores) == len(df_recent):
        df_recent["_compound"] = rec_scores
        most_pos = (
            df_recent.sort_values("_compound", ascending=False)
            .head(3)[["id","created_at","text","url","_compound"]]
            .to_dict(orient="records")
        )
        most_neg = (
            df_recent.sort_values("_compound", ascending=True)
            .head(3)[["id","created_at","text","url","_compound"]]
            .to_dict(orient="records")
        )
        mid = df_recent.iloc[(df_recent["_compound"].sub(0).abs()).sort_values().index].head(3)
        most_neu = mid[["id","created_at","text","url","_compound"]].to_dict(orient="records")
    else:
        most_pos = most_neg = most_neu = []

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": {
            "recent_rows": int(len(df_recent)),
            "baseline_rows": int(len(df_base))
        },
        "sentiment": {
            "mean_compound": mean_compound,
            "distribution_percent": {
                "very_positive": dist_pct.get("very_positive", 0.0),
                "positive":      dist_pct.get("positive", 0.0),
                "neutral":       dist_pct.get("neutral", 0.0),
                "negative":      dist_pct.get("negative", 0.0),
                "very_negative": dist_pct.get("very_negative", 0.0)
            }
        },
        "topics": {
            "rising_terms": rising,
            "evergreen_terms": evergreen,
            "top_bigrams": top_bi,
            "top_trigrams": top_tri
        },
        "examples": {
            "most_positive": most_pos,
            "most_negative": most_neg,
            "most_neutral":  most_neu
        }
    }

    (SITE / "nlp.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {SITE/'nlp.json'}")

if __name__ == "__main__":
    sys.exit(main())
