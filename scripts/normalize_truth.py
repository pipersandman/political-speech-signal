# scripts/normalize_truth.py
from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from datetime import datetime, timezone

TRUTH_REPO = os.getenv("TRUTH_REPO", "https://github.com/stiles/trump-truth-social-archive.git")
SNAP_MAX = int(os.getenv("SNAP_MAX", "200"))   # keep snapshot small & fast

def run(cmd, cwd=None):
    out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None)
    return out.decode("utf-8", "replace").strip()

def clone_repo(url: str, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    run(["git", "clone", "--depth=1", url, str(dest)])
    return dest

def json_records(p: Path):
    # Try JSONL/NDJSON first (one object per line)
    with p.open("rb") as f:
        first = f.read(1)
    if first == b"[":
        # JSON array
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    # JSONL
    rows = []
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def get(obj, *keys, default=None):
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            return obj[k]
    return default

def to_iso(dt):
    if isinstance(dt, str):
        return dt
    return None

def coerce(rec):
    # Flexible mapping: try common keys across archives
    rid = str(get(rec, "id", "status_id", "post_id", default=""))
    created = get(rec, "created_at", "published_at", "date", "time", default=None)
    text = get(rec, "content", "text", "message", "body", default="")
    url = get(rec, "url", "status_url", default=None)

    # simple metrics (default to 0 if missing)
    replies = int(get(rec, "replies_count", "reply_count", default=0) or 0)
    reposts = int(get(rec, "reblogs_count", "reblog_count", "repost_count", default=0) or 0)
    quotes  = int(get(rec, "quote_count", "quotes_count", default=0) or 0)
    likes   = int(get(rec, "favourites_count", "favorite_count", "favs_count", "like_count", default=0) or 0)

    author = get(rec, "account", "author", "user", default=None)
    if isinstance(author, dict):
        author = get(author, "acct", "username", "name", default=None)

    return {
        "id": rid,
        "created_at": to_iso(created),
        "source": "truth_social",
        "author": author,
        "text": text,
        "url": url,
        "metrics": {"replies": replies, "reposts": reposts, "quotes": quotes, "likes": likes},
        "raw": rec,  # keep original for later mapping improvements
    }

def main():
    # Clone upstream into a temp dir
    tmp = Path(tempfile.mkdtemp(prefix="truth_norm_"))
    repo_dir = clone_repo(TRUTH_REPO, tmp / "truth")

    # Heuristic: look for likely data files
    data_root = repo_dir / "data"
    candidates = []
    if data_root.exists():
        for p in data_root.rglob("*"):
            if p.suffix.lower() in {".json", ".jsonl", ".ndjson"} and p.name.lower().startswith(("truth", "posts", "archive")):
                candidates.append(p)
    else:
        candidates = [p for p in repo_dir.rglob("*") if p.suffix.lower() in {".json", ".jsonl", ".ndjson"}]

    # Read all records (lightweight), newest first if possible
    records = []
    for p in candidates:
        for rec in json_records(p):
            records.append(rec)

    # Coerce
    coerced = [coerce(r) for r in records]

    # Sort by created_at string desc if available
    def sort_key(r):
        v = r.get("created_at") or ""
        return v
    coerced.sort(key=sort_key, reverse=True)

    # Build snapshot (cap size)
    snap = coerced[:SNAP_MAX]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Stats
    total_seen = len(coerced)
    tmin = snap[-1]["created_at"] if snap else None
    tmax = snap[0]["created_at"] if snap else None

    site = Path("site")
    site.mkdir(parents=True, exist_ok=True)
    out = {
        "source_repo": TRUTH_REPO,
        "generated_at_utc": now,
        "total_seen": total_seen,
        "snapshot_size": len(snap),
        "time_range": {"min": tmin, "max": tmax},
        "records": snap,
    }
    (site / "snapshot.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {site/'snapshot.json'} with {len(snap)} rows (of {total_seen} seen)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
