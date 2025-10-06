from __future__ import annotations
import json, os, glob
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import spacy

# Reuse NLTK cache dir if set
nltk_data = os.environ.get("NLTK_DATA")
if nltk_data:
    nltk.data.path.append(nltk_data)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# Lightweight spaCy: NER + sentencizer only
nlp = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "textcat"])
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

SITE = "site"

def newest_parquet() -> str:
    files = sorted(glob.glob("data/processed/*.parquet"))
    if not files:
        raise SystemExit("[error] no parquet files found in data/processed/")
    return files[-1]

def main():
    path = newest_parquet()
    df = pd.read_parquet(path)

    # Choose a recent window (7d if timestamps; else newest 1000 rows)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    else:
        df["created_at"] = pd.NaT

    if df["created_at"].notna().any():
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=7)
        recent = df[df["created_at"] >= cutoff].copy()
        if len(recent) < 300:
            recent = df.sort_values("created_at", ascending=False).head(1000).copy()
    else:
        recent = df.head(1000).copy()

    # Compute post-level sentiment once
    texts = recent["text"].fillna("").astype(str).tolist()
    compounds = [sia.polarity_scores(t)["compound"] for t in texts]
    recent = recent.assign(_compound=compounds)

    # Extract entities and aggregate counts + sentiment
    ALLOWED = {"PERSON", "ORG", "GPE", "NORP", "LOC"}
    counts = defaultdict(int)
    sums = defaultdict(float)
    labels = {}

    for text, comp in zip(recent["text"].fillna("").astype(str), recent["_compound"]):
        if not text.strip():
            continue
        doc = nlp(text)
        seen_spans = set()
        for ent in doc.ents:
            if ent.label_ not in ALLOWED:
                continue
            key = ent.text.strip()
            # dedupe exact same span in this doc
            span_id = (ent.start_char, ent.end_char, ent.label_)
            if span_id in seen_spans:
                continue
            seen_spans.add(span_id)
            counts[(key, ent.label_)] += 1
            sums[(key, ent.label_)] += float(comp)
            labels[key] = ent.label_

    rows = []
    for (key, lab), c in counts.items():
        avg = sums[(key, lab)] / max(1, c)
        rows.append({"entity": key, "label": lab, "count": c, "avg_sentiment": round(avg, 3)})

    # Top lists by label
    def top(label, k=12):
        return [r for r in sorted(rows, key=lambda r: (r["count"], r["avg_sentiment"]), reverse=True) if r["label"] == label][:k]

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_size": int(len(recent)),
        "top_persons": top("PERSON"),
        "top_orgs": top("ORG"),
        "top_places": top("GPE") + top("LOC"),
        "top_groups": top("NORP")
    }

    os.makedirs(SITE, exist_ok=True)
    with open(os.path.join(SITE, "entities.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"[ok] wrote {SITE}/entities.json")

if __name__ == "__main__":
    main()
