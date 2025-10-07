import os, json, re, argparse, pathlib, csv, sys
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Silence HF pipeline warnings (cosmetic "sequential GPU" message)
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

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

def make_zeroshot():
    if not USE_TRANSFORMERS or not _TRANSFORMERS_OK:
        return None
    try:
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

def zeroshot_triage_batch(texts: List[str], threshold: float=0.5, batch_size: int=32) -> List[Dict[str,bool]]:
    global ZSC
    if ZSC is None:
        ZSC = make_zeroshot()
    if ZSC is None or not texts:
        return [ {} for _ in texts ]
    results = ZSC(
        sequences=texts,
        candidate_labels=ZSC_LABELS,
        multi_label=True,
        batch_size=batch_size,
        truncation=True,
        padding=True,
    )
    if isinstance(results, dict):
        results = [results]
    out = []
    for res in results:
        flags = {}
        for label, score in zip(res["labels"], res["scores"]):
            if score >= threshold:
                flags[ZSC_TO_KEY[label]] = True
        out.append(flags)
    return out

# -----------------------------
# Checkpointing helpers
# -----------------------------
def write_rows_append(csv_path: pathlib.Path, rows: List[Dict[str, Any]], header_fields: List[str], wrote_header_once: bool) -> bool:
    csv_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_fields)
        if not csv_exists and not wrote_header_once:
            w.writeheader()
            wrote_header_once = True
        for r in rows:
            w.writerow(r)
    return wrote_header_once

def load_checkpoint_ids(csv_path: pathlib.Path) -> set:
    if not csv_path.exists():
        return set()
    done = set()
    try:
        for row in pd.read_csv(csv_path, usecols=["id"]).itertuples(index=False):
            done.add(str(row.id))
    except Exception:
        pass
    return done

# -----------------------------
# Adjudication — BATCHED to JSON array
# -----------------------------
def load_schema_text():
    return (pathlib.Path(__file__).with_name("schema.json")).read_text(encoding="utf-8")

def make_user_prompt_multi(batch_posts: List[Dict[str, Any]]) -> str:
    schema = load_schema_text()
    lines = [
        "Return a JSON ARRAY. Each element must conform to this schema and include the same 'id' as provided.",
        "",
        schema,
        "",
        "POSTS:"
    ]
    for p in batch_posts:
        t = p["text"].replace("\n", " ")
        lines.append(f"id: {p['id']} | created_at: {p['created_at']} | text: {t}")
    return "\n".join(lines)

def adjudicate_batch_ollama(model: str, batch_posts: List[Dict[str, Any]], options: Dict[str, Any]) -> List[Dict[str, Any]]:
    user_prompt = make_user_prompt_multi(batch_posts)
    content = chat_ollama(model=model, system=ADJUDICATOR_SYSTEM_PROMPT, user=user_prompt, options=options)
    try:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array")
        by_id = {str(d.get("id")): d for d in data}
        rows = []
        for p in batch_posts:
            d = by_id.get(str(p["id"]))
            if not isinstance(d, dict):
                d = {k: False for k in CATEGORIES}
                d.update({
                    "targets_negative": [], "targets_positive": [],
                    "who_is_praised": [], "who_is_criticized": [],
                    "legal_action_details": "", "why": "LLM returned non-JSON or missing element; default false."
                })
            rows.append({**p, **d})
        return rows
    except Exception:
        rows = []
        for p in batch_posts:
            base = {k: False for k in CATEGORIES}
            base.update({
                "targets_negative": [], "targets_positive": [],
                "who_is_praised": [], "who_is_criticized": [],
                "legal_action_details": "", "why": "LLM returned invalid JSON; defaulting to triage."
            })
            rows.append({**p, **base})
        return rows

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
            if r.get(k): totals[k] += 1
        for t in r.get("who_is_praised", []):
            positive_targets[t] = positive_targets.get(t, 0) + 1
    return {"totals": totals, "positive_targets": positive_targets}

# -----------------------------
# Main (batched triage + batched adjudication + checkpoints)
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="File or directory of posts (json/jsonl/csv)")
    ap.add_argument("--outdir", default="./out", help="Output directory")

    ap.add_argument("--use_ollama", action="store_true", help="Use local Ollama model for adjudication")
    ap.add_argument("--ollama_model", default="llama3.1:8b", help="Ollama model name (ensure it's pulled)")
    ap.add_argument("--ollama_num_ctx", type=int, default=1024, help="Ollama num_ctx (lower = faster)")
    ap.add_argument("--ollama_num_batch", type=int, default=512, help="Ollama num_batch (higher can be faster)")
    ap.add_argument("--ollama_num_thread", type=int, default=0, help="Ollama CPU threads (0 = auto)")

    ap.add_argument("--adjudicate", choices=["all","triaged"], default="triaged", help="Which posts go to the LLM")
    ap.add_argument("--workers", type=int, default=4, help="Parallel batches for Ollama adjudication")
    ap.add_argument("--zsc_batch", type=int, default=64, help="Batch size for zero-shot triage")
    ap.add_argument("--zsc_threshold", type=float, default=0.5, help="Zero-shot flag threshold (0..1)")
    ap.add_argument("--skip_zeroshot", action="store_true", help="Skip zero-shot triage (heuristics only)")

    ap.add_argument("--adj_batch_size", type=int, default=12, help="Number of posts per Ollama adjudication call")
    ap.add_argument("--checkpoint_every", type=int, default=3, help="Write CSV every N completed batches")

    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "post_labels.csv"
    ckpt_jsonl = outdir / "partial_results.jsonl"

    # If resuming, collect already-done IDs
    done_ids = load_checkpoint_ids(csv_path)

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
                batch_flags = zeroshot_triage_batch(batch_texts, threshold=args.zsc_threshold, batch_size=args.zsc_batch)
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
        indices_for_llm = [i for i in range(len(posts)) if posts[i]["id"] not in done_ids]
    else:
        indices_for_llm = [i for i, flags in enumerate(tri_combined) if any(flags.values()) and posts[i]["id"] not in done_ids]

    # Prepare Ollama options
    ollama_options = {
        "temperature": 0,
        "num_ctx": args.ollama_num_ctx,
        "num_batch": args.ollama_num_batch,
        "num_thread": args.ollama_num_thread,
    }

    # Prepare CSV header and write any non-LLM rows immediately
    header_fields = list({
        "id": None, "created_at": None, "url": None, "text": None, **{k: None for k in CATEGORIES},
        "targets_negative": None, "targets_positive": None, "who_is_praised": None, "who_is_criticized": None,
        "legal_action_details": None, "why": None
    }.keys())
    wrote_header_once = csv_path.exists()

    # Write triage-only rows not going to LLM (and not already done)
    triage_only_rows = []
    for i, (p, flags) in enumerate(zip(posts, tri_combined)):
        if i not in indices_for_llm and p["id"] not in done_ids:
            data = {k: bool(flags.get(k, False)) for k in CATEGORIES}
            data.update({
                "targets_negative": [], "targets_positive": [],
                "who_is_praised": [], "who_is_criticized": [],
                "legal_action_details": "", "why": "Heuristic/triage-only (not sent to LLM)."
            })
            triage_only_rows.append({**p, **data})
    if triage_only_rows:
        wrote_header_once = write_rows_append(csv_path, triage_only_rows, header_fields, wrote_header_once)
        with open(ckpt_jsonl, "a", encoding="utf-8") as f:
            for r in triage_only_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Stage 4: Adjudication in BATCHES (parallel)
    batches = []
    for i in range(0, len(indices_for_llm), args.adj_batch_size):
        batch = indices_for_llm[i:i+args.adj_batch_size]
        batches.append(batch)

    completed_batches = 0
    to_write_rows: List[Dict[str, Any]] = []

    def work_on_batch(batch_idx_list: List[int]) -> List[Dict[str, Any]]:
        batch_posts = [posts[i] for i in batch_idx_list]
        return adjudicate_batch_ollama(args.ollama_model, batch_posts, ollama_options)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut2batch = {ex.submit(work_on_batch, b): b for b in batches}
        for fut in tqdm(as_completed(fut2batch), total=len(fut2batch), desc="Adjudication (batched + parallel)"):
            b = fut2batch[fut]
            try:
                rows = fut.result()
            except Exception as e:
                rows = []
                for i in b:
                    flags = tri_combined[i]
                    data = {k: bool(flags.get(k, False)) for k in CATEGORIES}
                    data.update({
                        "targets_negative": [], "targets_positive": [],
                        "who_is_praised": [], "who_is_criticized": [],
                        "legal_action_details": "",
                        "why": f"LLM error: {e}; falling back to triage."
                    })
                    rows.append({**posts[i], **data})

            # Append to buffer, write checkpoint periodically
            to_write_rows.extend(rows)
            completed_batches += 1
            if completed_batches % args.checkpoint_every == 0:
                wrote_header_once = write_rows_append(csv_path, to_write_rows, header_fields, wrote_header_once)
                with open(ckpt_jsonl, "a", encoding="utf-8") as f:
                    for r in to_write_rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                to_write_rows.clear()

    # Final flush
    if to_write_rows:
        wrote_header_once = write_rows_append(csv_path, to_write_rows, header_fields, wrote_header_once)
        with open(ckpt_jsonl, "a", encoding="utf-8") as f:
            for r in to_write_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Build the rollup from the checkpoint jsonl (covers everything written)
    all_rows = []
    if (outdir / "partial_results.jsonl").exists():
        with open(outdir / "partial_results.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_rows.append(json.loads(line))
                except:
                    pass

    report = aggregate(all_rows)
    (outdir/"oneoff_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# One-off Rhetoric Analysis Summary\n","## Totals\n"]
    for k, v in report["totals"].items():
        lines.append(f"- {k}: {v}")
    lines.append("\n## Positive Rhetoric — Who is praised (top 20)\n")
    for k, v in sorted(report["positive_targets"].items(), key=lambda kv: kv[1], reverse=True)[:20]:
        lines.append(f"- {k}: {v}")
    (outdir/"report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote (streaming): {csv_path}")
    print(f"Wrote (checkpoint stream): {outdir/'partial_results.jsonl'}")
    print(f"Wrote: {outdir/'oneoff_report.json'}")
    print(f"Wrote: {outdir/'report.md'}")

if __name__ == "__main__":
    main()
