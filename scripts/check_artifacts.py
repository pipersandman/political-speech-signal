# scripts/check_artifacts.py
import json, sys
from pathlib import Path

REQUIRED = {
    "site/inventory.json": ["source_repo","file_count","largest_file_bytes"],
    "site/snapshot.json":  ["total_seen","snapshot_size","time_range","records"],
    "site/trends.json":    ["generated_at_utc","days"],
    "site/nlp.json":       ["generated_at_utc","sentiment","topics","examples"],
}

def main():
    ok = True
    for path, keys in REQUIRED.items():
        p = Path(path)
        if not p.exists():
            print(f"[fail] missing {path}")
            ok = False
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[fail] bad json {path}: {e}")
            ok = False
            continue
        for k in keys:
            if k not in obj:
                print(f"[fail] {path} missing key: {k}")
                ok = False
    if ok:
        print("[ok] artifacts look good")
        return 0
    return 2

if __name__ == "__main__":
    sys.exit(main())
