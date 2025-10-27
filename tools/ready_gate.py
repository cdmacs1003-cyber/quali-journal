# tools/ready_gate.py
from typing import Callable, Dict, List, Tuple

Reason = str

def evaluate_ready(
    item: Dict,
    gate_required: int,
    id_is_unique: Callable[[str], bool],
    url_check: Callable[[str], Dict],
    allow_redirect_hops: int = 3
) -> Tuple[bool, List[Reason], Dict]:
    """
    Decide 'ready' from data-only rules (no UI re-judgement):
    1) approved == True
    2) value_score >= gate_required
    3) canonical_url resolves (HEAD/GET) within allowed redirects
    4) id is unique (system-wide)
    Returns: (ready, reasons, patch_fields)
    """
    reasons: List[Reason] = []
    patch: Dict = {}
    approved = bool(item.get("approved", False))
    if approved:
        reasons.append("approved")

    # score
    score = item.get("value_score", None)
    if score is not None and isinstance(score, (int, float)) and score >= gate_required:
        reasons.append("score>=gate")
    else:
        patch.setdefault("ready_debug", {})["score_detail"] = {"value_score": score, "gate": gate_required}

    # URL check
    url = (item.get("canonical_url") or "").strip()
    url_ok = False
    if url:
        chk = url_check(url)
        patch.update({
            "canonical_url": chk.get("final_url") or url,
            "url_status": chk.get("status", "dead"),
            "url_http_code": chk.get("http_code"),
            "url_redirect_hops": chk.get("hops", 0),
        })
        if chk.get("status") in ("ok", "redirect") and chk.get("hops", 0) <= allow_redirect_hops:
            url_ok = True
            reasons.append("url_ok")
    else:
        patch["url_status"] = "dead"

    # ID uniqueness
    id_val = item.get("id")
    unique_ok = bool(id_val) and id_is_unique(str(id_val))
    if unique_ok:
        reasons.append("id_unique")

    ready = (approved and
             (score is not None and score >= gate_required) and
             url_ok and
             unique_ok)

    patch["gate"] = gate_required
    patch["ready"] = ready
    patch["ready_reason"] = reasons
    return ready, reasons, patch
