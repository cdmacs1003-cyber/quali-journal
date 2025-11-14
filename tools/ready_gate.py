# tools/ready_gate.py
"""
Robust READY evaluator for QualiJournal.
- Handles dict and JSON-string items safely.
- Checks: approved, value_score >= gate_required, URL alive (with hop limit), ID uniqueness.
- Emits detailed debug fields when a check fails.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

# Type aliases
Reason = str
JSON = Dict[str, Any]
CheckFn = Callable[[str], JSON]
UniqueFn = Callable[[str], bool]


def _as_bool(value: Any) -> bool:
    """Coerce common truthy/falsy representations to bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "y", "yes", "true", "on", "ok", "approved"}


def _as_float(value: Any) -> Optional[float]:
    """Convert to float or return None on failure."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        return float(text) if text else None
    except Exception:
        return None


def _safe_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _normalise_item(item: Any, gate_required: int) -> Tuple[Optional[JSON], List[Reason], JSON]:
    """
    Ensure item is a dict. If it's a JSON string, try parse; else return not-ready with minimal patch.
    """
    if isinstance(item, dict):
        return item, [], {}
    if isinstance(item, str):
        try:
            loaded = json.loads(item)
            if isinstance(loaded, dict):
                return loaded, [], {}
        except Exception:
            pass
    # Invalid item → short-circuit with safe patch
    return None, ["invalid_item"], {"ready": False, "gate": gate_required}


def evaluate_ready(
    item: Any,
    gate_required: int,
    id_is_unique: UniqueFn,
    url_check: CheckFn,
    allow_redirect_hops: int = 3,
) -> Tuple[bool, List[Reason], JSON]:
    """
    Decide READY:
      1) approved == True
      2) value_score >= gate_required
      3) canonical_url alive (status in {'ok','redirect'}) AND hops ≤ allow_redirect_hops
      4) id uniqueness

    Returns: (ready: bool, reasons: [str], patch: dict)
    """
    # 0) Normalise input (dict or JSON-string)
    item, norm_reasons, norm_patch = _normalise_item(item, gate_required)
    if item is None:
        return False, norm_reasons, norm_patch

    reasons: List[Reason] = []
    patch: JSON = {}

    # 1) Approved
    approved = _as_bool(item.get("approved", False))
    if approved:
        reasons.append("approved")

    # 2) Score ≥ gate
    score = _as_float(item.get("value_score"))
    score_ok = (score is not None) and (score >= float(gate_required))
    if score_ok:
        reasons.append("score>=gate")
    else:
        patch.setdefault("ready_debug", {}).update({
            "score_detail": {"value_score": score, "gate": gate_required, "reason": "missing_or_below_gate"}
        })

    # 3) URL validity (+ hop limit)
    url = _safe_str(item.get("canonical_url"))
    url_ok = False
    if url:
        chk = url_check(url) or {}
        final_url = chk.get("final_url") or url
        status = chk.get("status", "dead")
        hops = int(chk.get("hops", 0) or 0)
        http_code = chk.get("http_code")

        patch.update({
            "canonical_url": final_url,
            "url_status": status,
            "url_http_code": http_code,
            "url_redirect_hops": hops,
        })

        if status in {"ok", "redirect"} and hops <= allow_redirect_hops:
            url_ok = True
            reasons.append("url_ok")
    else:
        patch["url_status"] = "dead"

    # 4) ID uniqueness
    id_val = item.get("id")
    unique_ok = False
    if id_val is not None:
        unique_ok = id_is_unique(_safe_str(id_val))
        if unique_ok:
            reasons.append("id_unique")

    # Final decision: all must pass
    ready = approved and score_ok and url_ok and unique_ok

    # Output
    patch["gate"] = gate_required
    patch["ready"] = ready
    patch["ready_reason"] = reasons
    return ready, reasons, patch
