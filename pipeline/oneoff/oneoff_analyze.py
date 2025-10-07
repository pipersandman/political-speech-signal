import os, json, re, argparse, pathlib, csv, sys, math
from typing import List, Dict, Any, Iterable, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# Triage with Transformers on GPU (batched)
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
# Zero-shot triage (BATCHED) on GPU if available
# -----------------------------
ZSC = None

def make_zeroshot():
    if not USE_TRANSFORMERS or not _TRANSFORMERS_OK:
        return None
    try:
        # Try CUDA:0 (RTX 4060). If not available, will fallback to CPU.
        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0
        )
    except Exception:
        try:
            return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception:
            return None

ZSC_LABELS = [
    "derogatory name-calling","violent rhetoric","angry tone","positive praise",
    "dehumanization","hyperbole","divisive labeling","call for unity","call for legal action"
]
ZSC_TO_KEY = {
    "derogatory name-calling":"derogatory_name_calling",
    "violent rhetoric":"violent_rhetoric",
    "angry tone":"angry_rhetoric",
    "positive praise":"positive_rhetoric",
    "dehumanization":"dehumanization",
    "hyperbole":"hyperbole",
    "divisive labeling":"divisive_labeling",
    "call for unity":"call_for_unity",
    "call for legal action":"call_for_legal_action",
}

def zeroshot_triage_batch(texts: List[str], threshold: float=0.5) -> List[Dict[str,bool]]:
    """Batch zero-shot triage for a list of texts. Returns list of dicts."""
    global ZSC
    if ZSC is None:
        ZSC = make_zeroshot()
    if ZSC is None or not texts:
        return [ {} for _ in texts ]

    # transformers supports batching via passing a list of sequences
    results = ZSC(
        sequences=texts,
        candidate_labels=ZSC_LABELS,
        multi_label=True
    )
    out = []
    # If a single item, pipeline returns dict not list
    if isinstance(results, dict):
        results = [results]
    for res in results:
        flags = {}
        for label, score in zip(res["labels"], res["scores"]):
            if score >= threshold:
                flags[ZSC_TO_KEY[label]] = True
        out.append(flags)
    return out

# -----------------------------
# LLM adjudication via Ollama (strict JSON), in parallel
# -----------------------------
def load_schema_text():
    return (pathlib.Path(__file__).with_name("schema.json")).read_text(encoding="utf-8")

def make_user_prompt(post_id: str, created_at: str, text: str) -> str:
    return ADJUDICATOR_USER_TEMPLATE.format(
        schema=load_schema_text(),
        post_id=post_id, created_at=created_at, text=text.replace("\n"," ")
    )

def ollama_adjudicate_single(
    model: str,
    post: Dict[str, Any],
    options: Dict[str, Any]
) -> Dict[str, Any]:
    user_prompt = make_user_prompt(post["id"], post["created_at"], post["text"])
    try:
        content = chat_ollama(model=model, system=ADJUDICATOR_SYSTEM_PROMPT, user=user_prompt, options=options)
        data = json.loads(content)
    except Exception:
        data = {k: False for k in CATEGORIES}
        data.update({
            "targets_negative": [], "targets_positive": [],
            "who_is_praised": [], "who_is_criticized": [],
            "legal_action_details": "", "why": "LLM returned non-JSON; default false."
        })
    return {**post, **data}

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
# Main (batched triage + parallel adjudication)
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="File or directory of posts (json/jsonl/csv)")
    ap.add_argument("--outdir", default="./out", help="Output directory")

    ap.add_argument("--use_ollama", action="store_true", help="Use local Ollama model for adjudication")
    ap.add_argument("--ollama_model", default="llama3.1:8b", help="Ollama model name (ensure it's pulled)")
    ap.add_argument("--ollama_num_ctx", type=int, default=1024, help="Ollama num_ctx (lower = faster)")
    ap.add_argument("--ollama_num_batch", type=int, default=512, help="Ollama num_batch (higher can be faster)")

    ap.add_argument("--adjudicate", choices=["all","triaged"], default="triaged", help="Which posts go to the LLM")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers for Ollama adjudication")
    ap.add_argument("--zsc_batch", type=int, default=32, help="Batch size for zero-shot triage")
    ap.add_argument("--zsc_threshold", type=float, default=0.5, help="Zero-shot flag threshold (0..1)")
    ap.add_argument("--skip_zeroshot", action="store_true", help="Skip zero-shot triage (heuristics only)")

    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    posts = load_posts(args.input)
    if not posts:
        print("No posts found.")
        sys.exit(1)

    # ---- Stage 1: Heuristics (fast)
    heur_flags = [triage_flags(p["text"]) for p in posts]

    # ---- Stage 2: Zero-shot triage in batches (GPU)
    zsc_flags = [{} for _ in posts]
    if not args.skip_zeroshot:
        global ZSC
        ZSC = make_zeroshot()
        if ZSC is not None:
            for i in tqdm(range(0, len(posts), args.zsc_batch), desc="Zero-shot triage (batched)"):
                batch_texts = [posts[j]["text"] for j in range(i, min(i+args.zsc_batch, len(posts)))]
                batch_flags = zeroshot_triage_batch(batch_texts, threshold=args.zsc_threshold)
                for k, f in enumerate(batch_flags):
                    zsc_flags[i+k] = f
        else:
            print("Zero-shot pipeline unavailable; proceeding with heuristics only.")

    # Combine triage flags
    tri_combined = []
    for h, z in zip(heur_flags, zsc_flags):
        combined = {k: bool(h.get(k, False) or z.get(k, False)) for k in CATEGORIES}
        tri_combined.append(combined)

    # ---- Stage 3: Decide which posts to send to LLM
    if args.adjudicate == "all":
        indices_for_llm = list(range(len(posts)))
    else:
        indices_for_llm = [i for i, flags in enumerate(tri_combined) if any(flags.values())]

    # ---- Stage 4: Run adjudication in parallel (Ollama)
    results: List[Optional[Dict[str, Any]]] = [None] * len(posts)

    # First, fill non-LLM rows with their triage-only decisions
    for i, (post, flags) in enumerate(zip(posts, tri_combined)):
        if i not in indices_for_llm:
            data = {k: bool(flags.get(k, False)) for k in CATEGORIES}
            data.update({
                "targets_negative": [], "targets_positive": [],
                "who_is_praised": [], "who_is_criticized": [],
                "legal_action_details": "", "why": "Heuristic/triage-only (not sent to LLM)."
            })
            results[i] = {**post, **data}

    ollama_options = {
        "temperature": 0,
        "num_ctx": args.ollama_num_ctx,
        "num_batch": args.ollama_num_batch,
    }

    if args.use_ollama and indices_for_llm:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {}
            for i in indices_for_llm:
                futures[ex.submit(ollama_adjudicate_single, args.ollama_model, posts[i], ollama_options)] = i
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Adjudication (parallel)"):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    # Fallback on error
                    flags = tri_combined[i]
                    data = {k: bool(flags.get(k, False)) for k in CATEGORIES}
                    data.update({
                        "targets_negative": [], "targets_positive": [],
                        "who_is_praised": [], "who_is_criticized": [],
                        "legal_action_details": "",
                        "why": f"LLM error: {e}; falling back to triage."
                    })
                    results[i] = {**posts[i], **data}
    else:
        # No LLM; everyone gets triage-only
        for i in indices_for_llm:
            flags = tri_combined[i]
            data = {k: bool(flags.get(k, False)) for k in CATEGORIES}
            data.update({
                "targets_negative": [], "targets_positive": [],
                "who_is_praised": [], "who_is_criticized": [],
                "legal_action_details": "", "why": "Heuristic-only (no LLM)."
            })
            results[i] = {**posts[i], **data}

    # Sanity: fill any Nones (shouldn't happen)
    for i, r in enumerate(results):
        if r is None:
            flags = tri_combined[i]
            data = {k: bool(flags.get(k, False)) for k in CATEGORIES}
            data.update({
                "targets_negative": [], "targets_positive": [],
                "who_is_praised": [], "who_is_criticized": [],
                "legal_action_details": "", "why": "Defaulted due to missing result."
            })
            results[i] = {**posts[i], **data}

    # ---- Outputs
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
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
