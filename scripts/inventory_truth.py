# scripts/inventory_truth.py
from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from datetime import datetime, timezone

TRUTH_REPO = os.getenv("TRUTH_REPO", "https://github.com/stiles/trump-truth-social-archive.git")
CACHE_BUST = os.getenv("CACHE_BUST", "")

def run(cmd, cwd=None):
    out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None)
    return out.decode("utf-8", "replace").strip()

def clone_repo(url: str, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    run(["git", "clone", "--depth=1", url, str(dest)])
    return dest

def file_linecount(p: Path) -> int:
    try:
        with p.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def main():
    site = Path("site")
    site.mkdir(parents=True, exist_ok=True)

    # Clone upstream for basic stats
    tmp = Path(tempfile.mkdtemp(prefix="truth_src_"))
    repo_dir = tmp / "truth"
    clone_repo(TRUTH_REPO, repo_dir)

    try:
        latest_commit_iso = run(["git", "log", "-1", "--format=%cI"], cwd=repo_dir)
    except Exception:
        latest_commit_iso = None

    data_root = repo_dir / "data"
    exts = {".jsonl", ".json", ".csv", ".tsv", ".ndjson"}
    files = []
    roots = [data_root] if data_root.exists() else [repo_dir]
    for root in roots:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)

    n_files = len(files)
    total_lines = 0
    largest = None
    largest_size = -1
    for p in files:
        total_lines += file_linecount(p)
        sz = p.stat().st_size
        if sz > largest_size:
            largest_size, largest = sz, p

    # Optional: read snapshot.json (created by normalize step)
    snap_path = site / "snapshot.json"
    snap = None
    if snap_path.exists():
        try:
            snap = json.loads(snap_path.read_text(encoding="utf-8"))
        except Exception:
            snap = None

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    inv = {
        "source_repo": TRUTH_REPO,
        "generated_at_utc": now,
        "latest_upstream_commit": latest_commit_iso,
        "file_count": n_files,
        "total_lines_approx": total_lines,
        "largest_file": str(largest.relative_to(repo_dir)) if largest else None,
        "largest_file_bytes": largest_size if largest_size >= 0 else None,
    }
    (site / "inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")

    # HTML
    snap_section = ""
    if snap:
        snap_section = f"""
  <div class="card">
    <h3>Today’s normalized snapshot</h3>
    <p>Total seen upstream: <b>{snap.get('total_seen')}</b> • Snapshot size: <b>{snap.get('snapshot_size')}</b></p>
    <p>Time range: <code>{(snap.get('time_range') or {}).get('min')}</code> → <code>{(snap.get('time_range') or {}).get('max')}</code></p>
    <p>Download: <a href="snapshot.json?v={CACHE_BUST}">snapshot.json</a></p>
  </div>
        """.strip()

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Political Speech Signal</title>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"/>
<meta http-equiv="Pragma" content="no-cache"/>
<meta http-equiv="Expires" content="0"/>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu;max-width:800px;margin:40px auto;padding:0 16px;line-height:1.5}}
code{{background:#f6f8fa;padding:2px 4px;border-radius:4px}}
.card{{border:1px solid #eee;border-radius:12px;padding:16px;margin:16px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.small{{color:#666}}
</style>
</head>
<body>
  <h1>Political Speech Signal</h1>
  <p class="small">Generated: {now}</p>

  <div class="card">
    <h3>Source inventory</h3>
    <p>Upstream repo: <code>{TRUTH_REPO}</code></p>
    <p>Latest upstream commit: <code>{latest_commit_iso or "unknown"}</code></p>
    <p>Data files discovered: <b>{n_files}</b></p>
    <p>Approx total lines: <b>{total_lines:,}</b></p>
    <p>Largest file: <code>{(largest.name if largest else "n/a")}</code> ({largest_size if largest_size>=0 else "n/a"} bytes)</p>
    <p>Raw inventory JSON: <a href="inventory.json?v={CACHE_BUST}">inventory.json</a></p>
  </div>

  {snap_section or ""}
  <p>Next in the waterfall: define the precise schema and add Parquet outputs.</p>
  <footer><small>© 2025</small></footer>
</body>
</html>"""
    (site / "index.html").write_text(html, encoding="utf-8")
    print(f"[ok] wrote {site/'index.html'} and {site/'inventory.json'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
