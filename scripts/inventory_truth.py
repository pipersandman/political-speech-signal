# scripts/inventory_truth.py
from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from datetime import datetime, timezone

TRUTH_REPO = os.getenv("TRUTH_REPO", "https://github.com/stiles/trump-truth-social-archive.git")

def run(cmd: list[str], cwd: Path | None = None) -> str:
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
    # where we’ll write the public site
    site = Path("site")
    site.mkdir(parents=True, exist_ok=True)

    # 1) clone the source repo into a temp dir on the Actions runner
    tmp = Path(tempfile.mkdtemp(prefix="truth_src_"))
    repo_dir = tmp / "truth"
    clone_repo(TRUTH_REPO, repo_dir)

    # 2) basic metadata from git
    try:
        latest_commit_iso = run(["git", "log", "-1", "--format=%cI"], cwd=repo_dir)
    except Exception:
        latest_commit_iso = None

    # 3) walk files and inventory likely data files
    data_root = repo_dir / "data"
    exts = {".jsonl", ".json", ".csv", ".tsv", ".ndjson"}
    files = []
    if data_root.exists():
        for p in data_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    else:
        # fallback: look everywhere
        for p in repo_dir.rglob("*"):
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

    # 4) produce a tiny inventory JSON (also shipped with the site)
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

    # 5) render a minimal index.html with the stats
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Political Speech Signal</title>
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
    <p>Raw inventory JSON: <a href="inventory.json">inventory.json</a></p>
  </div>
  <p>Next in the waterfall: define schema → normalize to Parquet → charts.</p>
  <footer><small>© 2025</small></footer>
</body>
</html>"""
    (site / "index.html").write_text(html, encoding="utf-8")

    print(f"[ok] wrote {site/'index.html'} and {site/'inventory.json'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
