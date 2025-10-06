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
    site = Path("site"); site.mkdir(parents=True, exist_ok=True)

    # Clone upstream just for inventory stats
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

    # Load snapshot/trends/NLP if present
    snap = None
    nlp = None
    trends = None
    s_path = site / "snapshot.json"
    n_path = site / "nlp.json"
    t_path = site / "trends.json"
    if s_path.exists():
        try: snap = json.loads(s_path.read_text(encoding="utf-8"))
        except Exception: pass
    if n_path.exists():
        try: nlp = json.loads(n_path.read_text(encoding="utf-8"))
        except Exception: pass
    if t_path.exists():
        try: trends = json.loads(t_path.read_text(encoding="utf-8"))
        except Exception: pass

    # HTML bits
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    snap_section = ""
    if snap:
        snap_section = f"""
  <div class="card">
    <h3>Today’s normalized snapshot</h3>
    <p>Total seen upstream: <b>{snap.get('total_seen')}</b> • Snapshot size: <b>{snap.get('snapshot_size')}</b></p>
    <p>Time range: <code>{(snap.get('time_range') or {}).get('min')}</code> → <code>{(snap.get('time_range') or {}).get('max')}</code></p>
    <p>Download: <a href="snapshot.json?v={CACHE_BUST}">snapshot.json</a></p>
  </div>"""

    nlp_section = ""
    if nlp:
        d = nlp.get("sentiment", {}).get("distribution_percent", {})
        vp, p, neu, n, vn = (d.get("very_positive",0), d.get("positive",0),
                             d.get("neutral",0), d.get("negative",0), d.get("very_negative",0))
        rising = nlp.get("topics", {}).get("rising_terms", [])[:8]
        ever = nlp.get("topics", {}).get("evergreen_terms", [])[:8]
        rising_html = " ".join(f"<code>{r['term']}</code>" for r in rising) or "n/a"
        ever_html = " ".join(f"<code>{r['term']}</code>" for r in ever) or "n/a"

        def bar(label, pct, color):
            return f'''
            <div style="display:grid;grid-template-columns:120px 1fr;gap:8px;align-items:center;margin:4px 0">
              <div style="color:#555">{label} {pct}%</div>
              <div style="background:#eee;border-radius:10px;overflow:hidden">
                <div style="height:10px;width:{pct}%;background:{color}"></div>
              </div>
            </div>'''.strip()

        bars = (
            bar("😀 very positive", vp, "#4caf50") +
            bar("🙂 positive",       p,  "#8bc34a") +
            bar("😐 neutral",        neu,"#9e9e9e") +
            bar("☹️ negative",      n,  "#ff9800") +
            bar("😠 very negative", vn, "#f44336")
        )

        nlp_section = f"""
  <div class="card">
    <h3>Sentiment & topics (recent window)</h3>
    {bars}
    <p style="margin-top:12px">Rising terms: {rising_html}</p>
    <p>Evergreen topics: {ever_html}</p>
    <p>Full NLP JSON: <a href="nlp.json?v={CACHE_BUST}">nlp.json</a></p>
  </div>"""


    trends_section = ""
    if trends:
        trends_section = f"""
  <div class="card">
    <h3>Trends (30d summary)</h3>
    <p><a href="trends.json?v={CACHE_BUST}">trends.json</a></p>
  </div>"""

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

  {snap_section}
  {nlp_section}
  {trends_section}

  <p>Next in the waterfall: richer NLP (entities), topic graphs, and a dashboard UI.</p>
  <footer><small>© 2025</small></footer>
</body>
</html>"""
    (site / "index.html").write_text(html, encoding="utf-8")
    print(f"[ok] wrote {site/'index.html'} and {site/'inventory.json'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
