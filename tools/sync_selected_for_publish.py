# tools/sync_selected_for_publish.py
import json, os, sys
from typing import Dict, List, Set, Tuple, Callable

# import from sibling tools
try:
    from tools.ready_gate import evaluate_ready
    from tools.url_check import check_url
except ImportError:
    # allow local import if run from project root
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ready_gate import evaluate_ready
    from url_check import check_url

CONFIG_PATH = os.environ.get("QUALI_CONFIG", "config.json")
DATA_DIR = os.environ.get("QUALI_DATA_DIR", "data")
ARTICLES_MAIN = os.path.join(DATA_DIR, "selected_articles.json")
ARTICLES_KEY = os.path.join(DATA_DIR, "selected_keyword_articles.json")

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

def _config_defaults() -> Dict:
    return {
        "gate_required": 70,
        "url_timeout_ms": 3000,
        "url_redirect_hops": 3
    }

def load_config() -> Dict:
    cfg = _config_defaults()
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try:
                file_cfg = json.load(f)
                if isinstance(file_cfg, dict):
                    cfg.update(file_cfg)
            except json.JSONDecodeError:
                pass
    return cfg

def build_id_uniqueness_checker(items: List[Dict]) -> Callable[[str], bool]:
    seen: Set[str] = set()
    def _check(x: str) -> bool:
        if x in seen:
            return False
        seen.add(x)
        return True
    return _check

def process(items: List[Dict], cfg: Dict, id_unique_check: Callable[[str], bool]) -> Tuple[List[Dict], Dict]:
    gate = int(cfg.get("gate_required", 70))
    hops = int(cfg.get("url_redirect_hops", 3))
    timeout_sec = max(0.1, float(cfg.get("url_timeout_ms", 3000)) / 1000.0)

    def url_checker(u: str):
        return check_url(u, timeout_sec=timeout_sec, max_hops=hops)

    ready_true = 0
    for it in items:
        ready, reasons, patch = evaluate_ready(it, gate, id_unique_check, url_checker, allow_redirect_hops=hops)
        # persist derived fields
        it.update(patch)
        if ready:
            ready_true += 1
    meta = {
        "total": len(items),
        "ready_true": ready_true,
        "ready_rate": (ready_true / len(items)) if items else 0.0,
        "gate_required": gate
    }
    return items, meta

def main():
    cfg = load_config()

    # load both lists and build a cross-set uniqueness checker
    items_main = load_json(ARTICLES_MAIN)
    items_key  = load_json(ARTICLES_KEY)

    # union for uniqueness
    union: List[Dict] = []
    if isinstance(items_main, list): union.extend(items_main)
    if isinstance(items_key, list):  union.extend(items_key)

    id_unique_check = build_id_uniqueness_checker([])  # we'll re-init per pass

    # First pass: ensure uniqueness across union
    seen: Set[str] = set()
    def idcheck(x: str) -> bool:
        nonlocal seen
        if x in seen:
            return False
        seen.add(x)
        return True

    # process each file independently but using shared uniqueness set
    items_main, meta_main = process(items_main or [], cfg, idcheck)
    items_key,  meta_key  = process(items_key  or [], cfg, idcheck)

    save_json(ARTICLES_MAIN, items_main)
    save_json(ARTICLES_KEY,  items_key)

    # print summary
    print(json.dumps({
        "main": meta_main,
        "keyword": meta_key
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
