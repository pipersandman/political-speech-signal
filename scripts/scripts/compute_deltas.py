# scripts/compute_deltas.py
from __future__ import annotations
import json, os, glob, re
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pandas as pd
import nltk, spacy
from nltk.sentiment import SentimentIntensityAnalyzer

# --- Setup lightweight NLP pieces ---
# NLTK cache (CI sets NLTK_DATA)
nltk_dir = os.environ.get("NLTK_DATA")
if nltk_dir:
    nltk.data.path.append(nltk_dir)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
try:
    from nltk.corpus import stopwords
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords

SIA = SentimentIntensityAnalyzer()
STOP = set(stopwords.words("english")) | {"rt","amp","https","http","t","co","www","com"}
URL_RE = re.compile(r"https?://\S+")
TOKEN_RE = re.compile(r"[#@]?[a-z0-9']{2,}")

# spaCy NER for entity deltas
NLP = spacy.load("en_core_web_sm", disable=["tagger","lemmatizer","textcat"])
if "sentencizer" not in NLP.pipe_names:
    NLP.add_pipe("sentencizer")
ALLOWED = {"NORP","ORG","GPE"}  # group-like entities

SITE = "site"

def newest_parquet() -> str:
    files = sorted(glob.glob("data/processed/*.parquet"))
    if not files:
        raise SystemExit("[error] no parquet files found in data/processed/")
    return files[-1]

def tokenize(txt: str):
    if not isinstance(txt, str): return []
    txt = URL_RE.sub(" ", txt.lower())
    toks = TOKEN_RE.findall(txt)
    out = []
    for t in toks:
        if t.startswith("@"):  # skip mentions
            continue
        if t.startswith("#"):  # strip hashtag
            t = t[1:]
        if not t or t in STOP or t.isdigit():
            continue
        out.append(t.strip("'"))
    return out

def tone_bucket(c: float) -> str:
    if c >= 0.6: return "very_positive"
    if c >= 0.2: return "positive"
    if c > -0.2: return "neutral"
    if c > -0.6: return "negative"
    return "very_negative"

def distrib(scores):
    buckets = Counter(tone_bucket(c) for c in scores)
    total = sum(buckets.values()) or 1
    return {k: round(100*b/total, 1) for k, b in buckets.items()}

def lift_terms(rec: Counter, base: Counter, k_min_recent=3, topk=8):
    vocab = set(rec) | set(base)
    V = max(1, len(vocab))
    R = sum(rec.values()) or 1
    B = sum(base.values()) or 1
    scored = []
    for t, r in rec.items():
        if r < k_min_recent: continue
        b = base.get(t, 0)
        rr = (r + 1) / (R + V)
        br = (b + 1) / (B + V)
        scored.append((t, r, b, rr/br))
    scored.sort(key=lambda x: (x[3], x[1]), reverse=True)
    return [{"term": t, "recent": r, "baseline": b, "lift": round(L, 2)} for t, r, b, L in scored[:topk]]

def entity_stats(texts: list[str]):
    """
    Sentence-level attribution: score each sentence, attribute to group-like entities in that sentence.
    Returns count and sentiment sum per canonical entity string.
    """
    counts = Counter()
    sums = defaultdict(float)
    for doc in NLP.pipe(texts, batch_size=64):
        for sent in doc.sents:
            s = sent.text.strip()
            if not s: continue
            comp = SIA.polarity_scores(s)["compound"]
            ents = {(e.text.strip(), e.label_) for e in sent.ents if e.label_ in ALLOWED}
            for etext, _ in ents:
                counts[etext] += 1
                sums[etext] += comp
    return counts, sums

def summarize_shift(mean_recent: float, mean_base: float, rising_terms: list[dict], top_entity: dict|None):
    delta = round(mean_recent - mean_base, 3)
    if delta <= -0.05:
        tone_txt = f"Tone leaned negative ({delta:+.2f} vs baseline)"
    elif delta >= 0.05:
        tone_txt = f"Tone leaned positive ({delta:+.2f} vs baseline)"
    else:
        tone_txt = f"Tone was steady ({delta:+.2f} vs baseline)"

    tops = ", ".join([f"“{r['term']}” {r['lift']}×" for r in rising_terms[:2]]) or "no clear risers"
    ent_txt = ""
    if top_entity:
        g = top_entity["entity"]
        ratio = top_entity["ratio"]
        s = top_entity["avg_sent_recent"]
        tone = "positive" if s >= 0.2 else ("negative" if s <= -0.2 else "mixed")
        ent_txt = f"; mentions of {g} were {ratio:.1f}× with {tone} sentiment ({s:+.2f})"
    return f"{tone_txt}. Rising terms: {tops}{ent_txt}."

def main():
    # --- Load data & windows ---
    path = newest_parquet()
    df = pd.read_parquet(path)
    if "text" not in df.columns:
        raise SystemExit("[error] parquet missing 'text' column")

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    else:
        df["created_at"] = pd.NaT

    now = pd.Timestamp.utcnow()
    cutoff_recent = now - pd.Timedelta(days=7)

    if df["created_at"].notna().any():
        recent = df[df["created_at"] >= cutoff_recent].copy()
        if len(recent) < 200:
            recent = df.sort_values("created_at", ascending=False).head(600).copy()
        cutoff_base = cutoff_recent - pd.Timedelta(days=180)
        base = df[(df["created_at"] < cutoff_recent) & (df["created_at"] >= cutoff_base)].copy()
        if len(base) < 1000:
            base = df.sort_values("created_at", ascending=True).head(5000).copy()
        recent_range = [str(recent["created_at"].min()), str(recent["created_at"].max())]
        base_range   = [str(base["created_at"].min()),   str(base["created_at"].max())]
    else:
        recent = df.head(600).copy()
        base = df.tail(5000).copy()
        recent_range = base_range = [None, None]

    r_texts = recent["text"].fillna("").astype(str).tolist()
    b_texts = base["text"].fillna("").astype(str).tolist()

    # --- Sentiment deltas (post-level, VADER) ---
    r_scores = [SIA.polarity_scores(t)["compound"] for t in r_texts] if r_texts else []
    b_scores = [SIA.polarity_scores(t)["compound"] for t in b_texts] if b_texts else []
    mean_r = round(sum(r_scores)/len(r_scores), 3) if r_scores else 0.0
    mean_b = round(sum(b_scores)/len(b_scores), 3) if b_scores else 0.0
    dist_r = distrib(r_scores)
    dist_b = distrib(b_scores)

    # --- Rising topics via lift (unigrams) ---
    r_uni = Counter(t for toks in map(tokenize, r_texts) for t in toks)
    b_uni = Counter(t for toks in map(tokenize, b_texts) for t in toks)
    rising = lift_terms(r_uni, b_uni, k_min_recent=3, topk=8)

    # --- Entity deltas (group-like) ---
    r_counts, r_sums = entity_stats(r_texts)
    b_counts, b_sums = entity_stats(b_texts)

    def top_mention_spikes():
        items = []
        for ent, rc in r_counts.items():
            bc = b_counts.get(ent, 0)
            if rc < 5:  # min recent mentions
                continue
            ratio = (rc + 1) / (bc + 1)
            r_avg = r_sums[ent] / max(1, rc)
            b_avg = (b_sums.get(ent, 0.0) / max(1, bc)) if bc else 0.0
            items.append({
                "entity": ent, "recent_mentions": int(rc), "baseline_mentions": int(bc),
                "ratio": round(ratio, 2),
                "avg_sent_recent": round(r_avg, 3),
                "avg_sent_baseline": round(b_avg, 3),
                "delta_sent": round(r_avg - b_avg, 3)
            })
        items.sort(key=lambda x: (x["ratio"], x["recent_mentions"]), reverse=True)
        return items[:8]

    mention_spikes = top_mention_spikes()
    top_ent = mention_spikes[0] if mention_spikes else None

    # --- Summary blurb ---
    blurb = summarize_shift(mean_r, mean_b, rising, top_ent)

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": {
            "recent_rows": int(len(recent)),
            "baseline_rows": int(len(base)),
            "recent_range": recent_range,
            "baseline_range": base_range
        },
        "sentiment": {
            "recent_mean": mean_r,
            "baseline_mean": mean_b,
            "delta": round(mean_r - mean_b, 3),
            "recent_dist_pct": dist_r,
            "baseline_dist_pct": dist_b
        },
        "topics": {
            "rising_terms": rising
        },
        "entities": {
            "mention_spikes": mention_spikes
        },
        "summary": blurb
    }

    os.makedirs(SITE, exist_ok=True)
    with open(os.path.join(SITE, "deltas.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"[ok] wrote {SITE}/deltas.json")

if __name__ == "__main__":
    main()
