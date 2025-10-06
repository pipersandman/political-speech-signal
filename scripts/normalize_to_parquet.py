from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

TRUTH_REPO = os.getenv("TRUTH_REPO", "https://github.com/stiles/trump-truth-social-archive.git")
TODAY = datetime.now(timezone.utc).date().isoformat()
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_OUT = OUT_DIR / f"utterances_{TODAY}.parquet"
SITE = Path("site"); SITE.mkdir(parents=True, exist_ok=True)

def run(cmd, cwd=None) -> str:
    out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None)
    return out.decode("utf-8", "replace").strip()

def clone_repo(url: str, dest: Path) -> Path:
    if dest.exists(): shutil.rmtree(dest)
    run(["git", "clone", "--depth=1", url, str(dest)])
    return dest

def json_records(p: Path):
    # Try array-JSON, else JSONL
    with p.open("rb") as f:
        first = f.read(1)
    if first == b"[":
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: return []
    rows = []
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

def get(obj, *keys, default=None):
    for k in keys:
        if isinstance(obj, dict) and k in obj and obj[k] not in (None,""):
            return obj[k]
    return default

def coerce(rec):
    rid   = str(get(rec, "id","status_id","post_id", default=""))
    c_at  = get(rec, "created_at","published_at","date","time", default=None)
    text  = get(rec, "content","text","message","body", default="")
    url   = get(rec, "url","status_url", default=None)
    author= get(rec, "account","author","user", default=None)
    if isinstance(author, dict):
        author = get(author, "acct","username","name", default=None)

    replies = int(get(rec, "replies_count","reply_count", default=0) or 0)
    reposts = int(get(rec, "reblogs_count","reblog_count","repost_count", default=0) or 0)
    quotes  = int(get(rec, "quote_count","quotes_count", default=0) or 0)
    likes   = int(get(rec, "favourites_count","favorite_count","favs_count","like_count", default=0) or 0)

    return {
        "id": rid,
        "created_at": c_at,
        "source": "truth_social",
        "author": author,
        "text": text,
        "url": url,
        "replies": replies,
        "reposts": reposts,
        "quotes": quotes,
        "likes": likes
    }

def main():
    tmp = Path(tempfile.mkdtemp(prefix="truth_parq_"))
    repo_dir = clone_repo(TRUTH_REPO, tmp / "truth")

    # Collect likely data files
    data_root = repo_dir / "data"
    candidates = []
    roots = [data_root] if data_root.exists() else [repo_dir]
    for root in roots:
        for p in root.rglob("*"):
            if p.suffix.lower() in {".json",".jsonl",".ndjson"} and p.name.lower().startswith(("truth","posts","archive")):
                candidates.append(p)

    rows = []
    for p in candidates:
        for rec in json_records(p):
            rows.append(coerce(rec))

    if not rows:
        print("[warn] no records found; skipping parquet write")
        return 0

    df = pd.DataFrame(rows)

    # Timestamps → pandas datetime
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    # Sort newest first, keep reasonable text dtype
    df = df.sort_values("created_at", ascending=False).reset_index(drop=True)

    # Write Parquet
    df.to_parquet(PARQUET_OUT, index=False)
    print(f"[ok] wrote {PARQUET_OUT} with {len(df):,} rows")

    # Tiny trend summary for the site (per-day count + mean likes)
    if df["created_at"].notna().any():
        per_day = (
            df.dropna(subset=["created_at"])
              .assign(day=lambda d: d["created_at"].dt.date.astype(str))
              .groupby("day")
              .agg(count=("id","count"), avg_likes=("likes","mean"))
              .reset_index()
              .tail(30)
        )
        trend = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "days": per_day.to_dict(orient="records")
        }
        (SITE / "trends.json").write_text(json.dumps(trend, indent=2), encoding="utf-8")
        print(f"[ok] wrote {SITE/'trends.json'} ({len(per_day)} days)")
    else:
        print("[warn] no valid timestamps; trends.json skipped")

    return 0

if __name__ == "__main__":
    sys.exit(main())
