# pipeline/enrich/extract_entities.py
"""
NER enrichment with progress bar, optional --head to test on a subset, and GPU acceleration.
Writes post_labels_enriched.csv next to the input CSV by default.
"""

import os
import re
import json
import argparse
import pandas as pd
import spacy

try:
    import torch
except Exception:
    torch = None

from tqdm import tqdm

MEDIA_KEYWORDS = {
    "cnn","fox","fox news","msnbc","nbc","abc","cbs","nytimes","new york times","washington post",
    "wapo","wsj","wall street journal","the atlantic","the guardian","bbc","reuters","ap","associated press",
    "newsmax","oan","one america","truth social","twitter","x.com","facebook","meta","instagram","tiktok",
    "press","media","fake news","journalist","reporter"
}

COURT_KEYWORDS = {
    "court","judge","justices","justice","scotus","supreme court","appeals court","da","district attorney",
    "doj","department of justice","fbi","attorney general","ag","special counsel","grand jury","prosecutor",
    "indictment","trial","jury","gag order","subpoena"
}

OPPONENT_BUCKET = {
    "joe biden","biden","kamala harris","harris","democrats","dems","the left","liberals","radical left",
    "hillary","obama","pelosi","schumer","romney","pence"
}

def normalize_text(s: str) -> str:
    s = s or ""
    return re.sub(r"\s+", " ", s.strip().lower())

def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    date_col = "created_at" if "created_at" in df.columns else ("date" if "date" in df.columns else None)
    if not date_col:
        raise ValueError("No 'created_at' or 'date' column in CSV.")
    df["created_at"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["created_at"]).sort_values("created_at")
    return df

def bucket_targets(raw_entities: list, raw_text: str) -> dict:
    text_norm = normalize_text(raw_text)
    media_hit = any(k in text_norm for k in MEDIA_KEYWORDS)
    court_hit = any(k in text_norm for k in COURT_KEYWORDS)
    opponents = set()
    for ent in raw_entities:
        t = normalize_text(ent["text"])
        if ent["label"] in {"PERSON","ORG","GPE"}:
            if t in OPPONENT_BUCKET:
                opponents.add(t)
            if any(tok in t for tok in ["democrat","democrats","the left","liberal","radical left","biden","harris"]):
                opponents.add(t)
    return {
        "media_targeted": int(media_hit),
        "court_targeted": int(court_hit),
        "opponents_targeted": int(len(opponents) > 0),
        "opponents_list": sorted(opponents),
    }

def move_transformer_to_cuda(nlp):
    """
    Move ONLY the transformer component to CUDA (no CuPy required).
    Works for both 'transformer' and 'curated_transformer' pipes, across spaCy versions.
    """
    if torch is None or not torch.cuda.is_available():
        print("[enrich] GPU not available; running on CPU.")
        return False

    # Pick the transformer pipe
    try:
        trf = nlp.get_pipe("transformer")
    except KeyError:
        try:
            trf = nlp.get_pipe("curated_transformer")
        except KeyError:
            print("[enrich] WARNING: No transformer pipe found; running CPU.")
            return False

    # Try common attributes that hold the torch module
    for obj in (
        getattr(trf, "model", None),
        getattr(trf, "torch", None),
        getattr(trf, "_model", None),
        trf,  # last resort if the pipe itself exposes .to()
    ):
        if obj is not None and hasattr(obj, "to"):
            try:
                obj.to("cuda")
                print(f"[enrich] Transformer moved to GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
                return True
            except Exception as e:
                print(f"[enrich] WARNING: attempt to move transformer to CUDA failed on {type(obj)}: {e}")

    print("[enrich] WARNING: could not locate a CUDA-movable module in transformer; running CPU.")
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="pipeline/oneoff/out/post_labels.csv",
                    help="Path to the base post_labels.csv")
    ap.add_argument("--out", default=None,
                    help="Output CSV (default: <input_dir>/post_labels_enriched.csv)")
    ap.add_argument("--model", default="en_core_web_trf",
                    help="spaCy model name, e.g. en_core_web_trf or en_core_web_md")
    ap.add_argument("--head", type=int, default=0,
                    help="If >0, only process the first N posts (for quick tests)")
    ap.add_argument("--batch_size", type=int, default=96,
                    help="spaCy pipe batch size (increase if you have GPU memory)")
    args = ap.parse_args()

    df = load_df(args.csv)
    if args.head and args.head > 0:
        df = df.head(args.head).copy()

    # Informational prints
    if torch is not None:
        try:
            print(f"[enrich] torch.cuda.is_available(): {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"[enrich] CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        except Exception:
            pass

    print(f"[enrich] loading spaCy model: {args.model}")
    try:
        nlp = spacy.load(args.model)
    except OSError:
        raise SystemExit(
            f"spaCy model '{args.model}' not found. Install with:\n"
            f"    python -m spacy download {args.model}\n"
            f"or pip install the wheel directly."
        )

    # >>> Move only the transformer to CUDA (no CuPy needed)
    moved = move_transformer_to_cuda(nlp)
    if not moved:
        print("[enrich] Proceeding on CPU for transformer.")

    texts = df["text"].fillna("").astype(str).tolist()
    print(f"[enrich] NER over {len(texts)} posts … (batch_size={args.batch_size})")

    raw_json = []
    media_flags, court_flags, opp_flags, opp_lists = [], [], [], []

    for i, doc in enumerate(tqdm(nlp.pipe(texts, batch_size=args.batch_size), total=len(texts), desc="NER")):
        ents = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        raw_json.append(json.dumps(ents, ensure_ascii=False))
        buckets = bucket_targets(ents, texts[i])
        media_flags.append(buckets["media_targeted"])
        court_flags.append(buckets["court_targeted"])
        opp_flags.append(buckets["opponents_targeted"])
        opp_lists.append(", ".join(buckets["opponents_list"]))

        if (i + 1) % 1000 == 0:
            print(f"[enrich] processed {i+1}/{len(texts)}")

    df["ner_entities"] = raw_json
    df["attacks_media"] = media_flags
    df["attacks_courts"] = court_flags
    df["attacks_political_opponents"] = opp_flags
    df["opponents_list"] = opp_lists

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "post_labels_enriched.csv")
    df.to_csv(out_path, index=False)
    print(f"[enrich] wrote: {out_path}")

if __name__ == "__main__":
    main()
