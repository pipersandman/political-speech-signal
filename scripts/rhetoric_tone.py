from __future__ import annotations
import json, os, glob, re
from collections import defaultdict, Counter
from datetime import datetime, timezone

import pandas as pd
import nltk, spacy
from nltk.sentiment import SentimentIntensityAnalyzer

# Reuse cached NLTK if present
nltk_dir = os.environ.get("NLTK_DATA")
if nltk_dir:
    nltk.data.path.append(nltk_dir)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# spaCy model (installed in CI)
nlp = spacy.load("en_core_web_sm", disable=["tagger","lemmatizer","textcat"])
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

SITE = "site"

PRAISE = {
    "great","strong","brave","patriot","respect","love","win","winning",
    "tremendous","beautiful","amazing","incredible","genius","courage"
}
ATTACK = {
    "corrupt","crooked","rigged","weak","fake","liar","disaster","enemy",
    "radical","extremist","thugs","terrible","stupid","traitor","crime","illegal"
}

EXCLAM = re.compile(r"!{1,}")
ALLCAP = re.compile(r"\b[A-Z]{4,}\b")

CANON_MAP = {
    "democrats":"Democrats","dems":"Democrats","democrat":"Democrats",
    "republicans":"Republicans","gop":"Republicans","republican":"Republicans",
    "media":"Media","press":"Media","mainstream media":"Media","fake news":"Media",
    "immigrants":"Immigrants","migrants":"Immigrants","illegals":"Immigrants",
    "china":"China","ccp":"China","chinese":"China",
    "mexico":"Mexico","mexicans":"Mexico",
    "israel":"Israel","palestinians":"Palestinians","palestinian":"Palestinians",
    "muslims":"Muslims","christians":"Christians","jews":"Jews"
}

ALLOWED = {"NORP","ORG","GPE"}  # group-like entities

def newest_parquet() -> str:
    files = sorted(glob.glob("data/processed/*.parquet"))
    if not files:
        raise SystemExit("[error] no parquet files in data/processed/")
    return files[-1]

def canon_label(text: str, label: str) -> str:
    t = text.lower().strip()
    if t in CANON_MAP:
        return CANON_MAP[t]
    return text.strip()

def tone_bucket(c: float) -> str:
    if c >= 0.6:  return "very_positive"
    if c >= 0.2:  return "positive"
    if c > -0.2:  return "neutral"
    if c > -0.6:  return "negative"
    return "very_negative"

def intensity_feats(sent_text: str) -> dict:
    toks = set(w.lower().strip(".,!?;:") for w in sent_text.split())
    return {
        "praise": len(PRAISE & toks),
        "attack": len(ATTACK & toks),
        "exclam": 1 if EXCLAM.search(sent_text) else 0,
        "allcap": 1 if ALLCAP.search(sent_text) else 0
    }

def main():
    path = newest_parquet()
    df = pd.read_parquet(path)

    # time window: last 7d if timestamps; else newest 1500 rows
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=7)
        recent = df[df["created_at"] >= cutoff]
        if len(recent) < 400:
            recent = df.sort_values("created_at", ascending=False).head(1500)
    else:
        recent = df.head(1500)
    recent = recent.fillna({"text": ""})

    group_counts = Counter()
    group_sent_sum = defaultdict(float)
    group_bucket = defaultdict(Counter)
    group_intensity = defaultdict(lambda: {"praise":0,"attack":0,"exclam":0,"allcap":0})

    for text in recent["text"].astype(str):
        if not text.strip():
            continue
        doc = nlp(text)
        for sent in doc.sents:
            s = sent.text.strip()
            if not s:
                continue
          comp = sia.polarity_scores(s)["compound"]  # reuse the global SIA
feats = intensity_feats(s)

# keep only group-like entities and dedupe them
ents = [e for e in sent.ents if e.label_ in ALLOWED]
uniq = {(e.text, e.label_) for e in ents}

for etext, elab in uniq:
    canon = canon_label(etext, elab)
    group_counts[canon] += 1
    group_sent_sum[canon] += comp
    group_bucket[canon][tone_bucket(comp)] += 1
    gi = group_intensity[canon]
    gi["praise"] += feats["praise"]
    gi["attack"] += feats["attack"]
    gi["exclam"] += feats["exclam"]
    gi["allcap"] += feats["allcap"]


    rows = []
    for g, c in group_counts.items():
        avg = group_sent_sum[g] / max(1, c)
        rows.append({
            "group": g,
            "mentions": c,
            "avg_sentiment": round(avg, 3),
            "distribution": dict(group_bucket[g]),
            "intensity": group_intensity[g]
        })

    MIN_M = 5
    praised = sorted([r for r in rows if r["mentions"] >= MIN_M],
                     key=lambda r: (r["avg_sentiment"], r["mentions"]), reverse=True)[:10]
    criticized = sorted([r for r in rows if r["mentions"] >= MIN_M],
                        key=lambda r: (r["avg_sentiment"], -r["mentions"]))[:10]

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_rows": int(len(recent)),
        "groups": sorted(rows, key=lambda r: r["mentions"], reverse=True)[:100],
        "top_praised": praised,
        "top_criticized": criticized
    }

    os.makedirs(SITE, exist_ok=True)
    with open(os.path.join(SITE, "rhetoric.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"[ok] wrote {SITE}/rhetoric.json")

if __name__ == "__main__":
    main()
