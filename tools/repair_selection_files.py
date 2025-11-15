# tools/repair_selection_files.py
"""
One-off backfill script:
- adds 'ready', 'ready_reason', 'gate', 'url_status' if missing
- re-evaluates current items and saves back
Safe to re-run; always persists derived fields into JSON.
"""
import json, os, sys
from typing import Dict, List, Set

try:
    from tools.ready_gate import evaluate_ready
    from tools.url_check import check_url
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ready_gate import evaluate_ready
    from url_check import check_url

CONFIG_PATH = os.environ.get("QUALI_CONFIG", "config.json")
DATA_DIR = os.environ.get("QUALI_DATA_DIR", "data")
FILES = [
    os.path.join(DATA_DIR, "selected_articles.json"),
    os.path.join(DATA_DIR, "selected_keyword_articles.json"),
]

def load_json(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_config():
    cfg = {
        "gate_required": 70,
        "url_timeout_ms": 3000,
        "url_redirect_hops": 3
    }
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try:
                file_cfg = json.load(f)
                if isinstance(file_cfg, dict):
                    cfg.update(file_cfg)
            except json.JSONDecodeError:
                pass
    return cfg

def main():
    cfg = load_config()
    gate = int(cfg.get("gate_required", 70))
    hops = int(cfg.get("url_redirect_hops", 3))
    timeout_sec = max(0.1, float(cfg.get("url_timeout_ms", 3000)) / 1000.0)

    def url_checker(u: str):
        return check_url(u, timeout_sec=timeout_sec, max_hops=hops)

    # load union → id uniqueness
    union: List[Dict] = []
    for p in FILES:
        union.extend(load_json(p) or [])
    seen: Set[str] = set()
    def id_is_unique(x: str) -> bool:
        nonlocal seen
        if x in seen:
            return False
        seen.add(x)
        return True

    for p in FILES:
        items = load_json(p) or []
        for it in items:
            _, _, patch = evaluate_ready(it, gate, id_is_unique, url_checker, allow_redirect_hops=hops)
            # --- 안전 업데이트: 문자열/기타 타입 방어 ---
            try:
                _obj = it
                if isinstance(_obj, str):
                    try:
                        _obj = json.loads(_obj)
                    except Exception:
                        _obj = None
                if isinstance(_obj, dict):
                    _obj.update(patch)
                    it = _obj
                # dict가 아니면 패스(해당 항목은 건너뜀)
            except Exception:
                pass

        save_json(p, items)

    print("Backfill OK.")

if __name__ == "__main__":
    main()

