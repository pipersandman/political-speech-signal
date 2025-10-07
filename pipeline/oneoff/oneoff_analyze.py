import os, json, re, argparse, pathlib, csv, sys
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Optional triage with Transformers (zero-shot), on GPU if available
# -----------------------------
USE_TRANSFORMERS = True
try:
    from transformers import pipeline
    _TRANSFORMERS_OK = True
except Exception:
    _TRANSFORMERS_OK = False
    USE_TRANSFORMERS = False

# -----------------------------
# Local LLM adjudication via Ollama
# -----------------------------
from prompts import ADJUDICATOR_SYSTEM_PROMPT, ADJUDICATOR_USER_TEMPLATE
from ollama_client import chat_ollama

CATEGORIES = [
    "derogatory_name_calling",
    "violent_rhetoric",
    "angry_rhetoric",
    "positive_rhetoric",
    "dehumanization",
    "hyperbole",
    "divisive_labeling",
    "call_for_unity",
    "call_for_legal_action",
]

# ---- Heuristic triage patterns (just to route; not source of truth)
HYPERBOLE_PATTERNS = [
    r"\b(best|greatest|biggest|worst|strongest)\b.*\b(ever|in history|of all time)\b",
    r"\b(total|complete|absolute)\s+(disaster|failure|victory|lie)\b",
    r"\bnever\b.*\b(seen|done|happened)\b",
]
UNITY_PATTERNS = [ r"\b(unite|unity|come together|all americans|both sides|work together)\b" ]
LEGAL_ACTION_PATTERNS = [ r"\b(indict|indictment|prosecute|prosecution|charge|charged|jail|prison|lock\s*(him|her|them)?\s*up)\b" ]
DIVISIVE_LABELLING_TERMS = ["rino","radicals","enemy of the people","deep state","globalists","the media","the elites"]
DEROGATORY_CUES = ["loser","coward","idiot","clown","disgrace"]
VIOLENT_CUES   = ["attack","crush","destroy","fight","war","violence"]
POSITIVE_CUES  = ["great","tremendous","wonderful","love","thank","honor"]
ANGRY_CUES     = ["furious","angry","rage","disgusting","outrage"]
DEHUMANIZATION_CUES = ["animals","vermin","rats","infestation","plague","subhuman"]

def any_pattern(text, patterns):
    return any(re.search(p, text, flags=re.I) for p in patterns)

def any_term(text, terms):
    t = text.lower()
    return any(term in t for term in terms)

def triage_flags(text: str) -> Dict[str,bool]:
    return {
        "hyperbole": any_pattern(text, HYPERBOLE_PATTERNS),
        "call_for_unity": any_pattern(text, UNITY_PATTERNS),
        "call_for_legal_action": any_pattern(text, LEGAL_ACTION_PATTERNS),
        "divisive_labeling": any_term(text, DIVISIVE_LABELLING_TERMS),
        "derogatory_name_calling": any_term(text, DEROGATORY_CUES),
        "violent_rhetoric": any_term(text, VIOLENT_CUES),
        "positive_rhetoric": any_term(text, POSITIVE_CUES),
        "angry_rhetoric": any_term(text, ANGRY_CUES),
        "dehumanization": any_term(text, DEHUMANIZATION_CUES),
    }

# -----------------------------
# Zero-shot triage on GPU (CUDA:0) if available
# -----------------------------
def make_zeroshot():
    if not USE_TRANSFORMERS or not _TRANSFORMERS_OK:
        return None
    try:
        # Try to place on CUDA device 0 if available
        # If CUDA isn't available, transformers will fall back to CPU automatically
        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0  # CUDA:0 for RTX 4060; falls back to CPU if no CUDA
        )
    except Exception:
        # Graceful fallback (CPU)
        try:
            return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception:
            return None

ZSC = None

def zeroshot_triage(text: str) -> Dict[str,bool]:
    """Returns dict of potential labels from zero-shot triage (thresholded), otherwise {}."""
    global ZSC
    if ZSC is None:
        ZSC = make_zeroshot()
    if ZSC is None:
        return {}
    labels = [
        "derogatory name-calling","violent rhetoric","angry tone","positive praise",
        "dehumanization","hyperbole","divisive labeling","call for unity","call for legal action"
    ]
    res = ZSC(text, candidate_labels=labels, multi_label=True)
    out = {}
    # threshold ~0.5 for a "maybe"
    for label, score in zip(res["labels"], res["scores"]):
        if score >= 0.5:
            key = {
                "derogatory name-calling":"derogatory_name_calling",
                "violent rhetoric":"violent_rhetoric",
                "angry tone":"angry_rhetoric",
                "positive praise":"positive_rhetoric",
                "dehumanization":"dehumanization",
                "hyperbole":"hyperbole",
                "divisive labeling":"divisive_labeling",
                "call for unity":"call_for_unity",
                "call for legal action":"call_for_legal_action",
            }[label]
            out[key] = True
    return out

# -----------------------------
# LLM adjudication via Ollama (strict JSON)
# -----------------------------
def load_schema_text():
    return (pathlib.Path(__file__).with_name("schema.json")).read_text(encoding="utf-8")

def ollama_adjudicate(model: str, text: str, post_id: str, created_at: str) -> Dict[str, Any]:
    user_prompt = ADJUDICATOR_USER_TEMPLATE.format(
        schema=load_schema_text(),
        post_id=post_id, created_at=created_at, text=text.replace("\n"," ")
    )
    content = chat_ollama(model=model, system=ADJUDICATOR_SYSTEM_PROMPT, user=user_prompt)
    try:
        return json.loads(content)
    except Exception:
        base = {k: False for k in CATEGORIES}
        base.update({
            "targets_negative": [], "targets_positive": [],
            "who_is_praised": [], "who_is_criticized": [],
            "legal_action_details": "", "why": "LLM returned non-JSON; default false."
        })
        return base

# -----------------------------
# Data loading helpers
# -----------------------------
def coalesce_text(r):
    for k in ["text","content","body","message"]:
        if k in r and isinstance(r[k], str) and r[k].strip():
            return r[k]
    return ""

def coalesce_id(r, i):
    for k in ["id","post_id","uuid"]:
        if k in r and str(r[k]).strip():
            return str(r[k])
    return f"row_{i}"

def coalesce_created_at(r):
    for k in ["created_at","published_at","date","timestamp"]:
        if k in r and str(r[k]).strip():
            return str(r[k])
    return ""

def load_any(path: pathlib.Path) -> List[Dict[str,Any]]:
    items = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            try: items.append(json.loads(line))
            except: pass
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list): items += data
        elif isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list): items += data["data"]
            else: items.append(data)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        items += df.to_dict(orient="records")
    return items

def load_posts(input_path: str) -> List[Dict[str,Any]]:
    p = pathlib.Path(input_path)
    files = [p] if p.is_file() else [fp for ext in ("*.jsonl","*.json","*.csv") for fp in p.rglob(ext)]
    records = []
    for fp in files:
        records += load_any(fp)
    posts = []
    for i, r in enumerate(records):
        text = coalesce_text(r)
        if not text.strip(): continue
        posts.append({
            "id": coalesce_id(r, i),
            "created_at": coalesce_created_at(r),
            "url": r.get("url",""),
            "text": text.strip()
        })
    return posts

# -----------------------------
# Aggregation/report
# -----------------------------
def aggregate(rows: List[Dict[str,Any]]) -> Dict[str,Any]:
    totals = {k: 0 for k in CATEGORIES}
    positive_targets = {}
    for r in rows:
        for k in CATEGORIES:
            if r[k]: totals[k] += 1
        for t in r.get("who_is_praised", []):
            positive_targets[t] = positive_targets.get(t, 0) + 1
    return {"totals": totals, "positive_targets": positive_targets}

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="File or directory of posts (json/jsonl/csv)")
    ap.add_argument("--outdir", default="./out", help="Output directory")
    ap.add_argument("--use_ollama", action="store_true", help="Use local Ollama model for adjudication")
    ap.add_argument("--ollama_model", default="llama3.1:8b", help="Ollama model name (ensure it's pulled)")
    ap.add_argument("--adjudicate", choices=["all","triaged"], default="triaged", help="Which posts go to the LLM")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    posts = load_posts(args.input)
    if not posts:
        print("No posts found.")
        sys.exit(1)

    results = []
    for post in tqdm(posts, desc="Analyzing"):
        text = post["text"]

        # Heuristic triage + zero-shot triage
        tri = triage_flags(text)
        ztri = zeroshot_triage(text)
        tri_combined = {k: tri.get(k, False) or ztri.get(k, False) for k in CATEGORIES}

        do_llm = (args.adjudicate == "all") or any(tri_combined.values())
        if args.use_ollama and do_llm:
            data = ollama_adjudicate(args.ollama_model, text, post_id=post["id"], created_at=post["created_at"])
        else:
            data = {k: bool(tri_combined.get(k, False)) for k in CATEGORIES}
            data.update({
                "targets_negative": [], "targets_positive": [],
                "who_is_praised": [], "who_is_criticized": [],
                "legal_action_details": "", "why": "Heuristic-only (no LLM)."
            })

        results.append({**post, **data})

    # Outputs
    csv_path = outdir / "post_labels.csv"
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)

    report = aggregate(results)
    (outdir/"oneoff_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# One-off Rhetoric Analysis Summary\n","## Totals\n"]
    for k, v in report["totals"].items():
        lines.append(f"- {k}: {v}")
    lines.append("\n## Positive Rhetoric — Who is praised (top 20)\n")
    for k, v in sorted(report["positive_targets"].items(), key=lambda kv: kv[1], reverse=True)[:20]:
        lines.append(f"- {k}: {v}")
    (outdir/"report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {outdir/'oneoff_report.json'}")
    print(f"Wrote: {outdir/'report.md'}")

if __name__ == "__main__":
    main()
