from qj_paths import rel as qj_rel
"""Common utility functions for QualiNews.

This module centralizes helper functions and configuration utilities that were
previously duplicated across multiple components such as ``orchestrator.py``
and ``engine_core.py``. By importing from this shared module, the rest of
the codebase can avoid copying logic and maintain a single source of truth.

Functions included:

* :func:`canonical_url` ??normalizes URLs by stripping fragments and trailing
  slashes.
* :func:`deep_merge` ??recursively merge two dictionaries, used for
  combining user configuration with defaults.
* :func:`split_keyword_tokens` ??tokenizes a keyword while preserving
  separators used in standard numbers (``-``, ``/``, ``+`` and ``.``) and
  discarding one?멵haracter alphabetic tokens.
* :func:`safe_compile_regex_list` ??compiles a list of regular expressions
  while ignoring patterns with syntax errors.
* :func:`norm01_log` ??logarithmically normalize a value between 0 and 1
  given a maximum value.
* :func:`value_score` ??compute a simple score based on keyword hits,
  upvotes and views using configurable weights and normalization factors.

These functions were originally defined in ``orchestrator.py`` and are
intended to be shared across modules. Import and use them instead of
maintaining local copies.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Pattern


def canonical_url(u: str) -> str:
    """Return a canonicalized version of the URL.

    This helper removes URL fragments (``#??`) and trailing slashes. If the
    input is falsy, an empty string is returned.

    Args:
        u: URL string to normalize.

    Returns:
        Normalized URL string.
    """
    try:
        u = (u or "").strip()
        # Drop URL fragment (after '#')
        u = re.sub(r"#.*$", "", u)
        # Remove trailing slash(es)
        u = re.sub(r"/+\s*$", "", u)
        return u
    except Exception:
        return u or ""


def deep_merge(a: dict | None, b: dict | None) -> dict:
    """Recursively merge two dictionaries.

    A convenience wrapper around a simple dict merge that recurses into
    nested dictionaries. Keys from ``b`` override those in ``a``. If either
    input is not a mapping, it is treated as an empty dict.

    Args:
        a: Base dictionary. Will not be modified.
        b: Overlay dictionary whose values override those in ``a``.

    Returns:
        A new dictionary representing the merged result.
    """
    if not isinstance(a, dict):
        a = {}
    if not isinstance(b, dict):
        return dict(a)
    out = dict(a)
    for k, v in b.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = deep_merge(out.get(k, {}), v)
        else:
            out[k] = v
    return out


def split_keyword_tokens(kw: str) -> List[str]:
    """Split a keyword string into tokens while preserving standard separators.

    The splitting logic uses non?멲lphanumeric characters as delimiters but
    preserves ``-``, ``/``, ``+`` and ``.`` because standards such as
    ``IPC-A-610`` include these as meaningful parts. Single?멿etter alphabetic
    tokens (e.g. the ``A`` in ``IPC-A-610``) are discarded to avoid noise.

    Args:
        kw: Raw keyword string.

    Returns:
        A list of lowercase tokens.
    """
    if not kw:
        return []
    # Split on any char that is not word, slash, dash, plus or dot
    toks = re.split(r"[^\w\-/+\.]+", kw)
    return [t.lower() for t in toks if t and (t.isdigit() or len(t) >= 2)]


def safe_compile_regex_list(patterns: Optional[List[str]]) -> List[Pattern]:
    """Compile a list of regular expressions, ignoring invalid ones.

    Patterns are compiled case?멼nsensitively. If a pattern fails to compile,
    it is silently skipped.

    Args:
        patterns: List of regex pattern strings or None.

    Returns:
        List of compiled regex patterns.
    """
    out: List[Pattern] = []
    for p in patterns or []:
        try:
            out.append(re.compile(p, re.I))
        except re.error:
            continue
    return out


def norm01_log(x: int | float, maxv: int | float) -> float:
    """Normalize a value logarithmically into the [0,1] range.

    When ``maxv`` is non?몆ositive, returns 0.0. Otherwise computes
    ``log1p(max(0,x)) / log1p(maxv)`` and clamps to 1.0.

    Args:
        x: Value to normalize.
        maxv: Maximum value corresponding to 1.0.

    Returns:
        Normalized float between 0.0 and 1.0.
    """
    if maxv <= 0:
        return 0.0
    return min(1.0, math.log1p(max(0.0, float(x))) / math.log1p(float(maxv)))


def value_score(kw_raw: int, upvotes: int, views: int, cfg: Dict) -> float:
    """Compute a simple relevance score for a community article.

    The score is a weighted sum of keyword hits, upvotes and views. Weights
    and normalization parameters are read from the provided config dict.

    Args:
        kw_raw: Number of keyword hits (raw count).
        upvotes: Number of upvotes.
        views: Number of views.
        cfg: Config dictionary containing ``score_weights`` and ``norms``.

    Returns:
        A floating?몆oint score.
    """
    c = cfg.get("community", {}) if isinstance(cfg, dict) else {}
    W = c.get("score_weights", {"keyword": 3, "upvotes": 5, "views": 2})
    N = c.get("norms", {"kw_base": 2, "upvotes_max": 200, "views_max": 100000})
    kw_norm = min(1.0, (kw_raw or 0) / max(1, int(N.get("kw_base", 2))))
    up_norm = norm01_log(upvotes or 0, int(N.get("upvotes_max", 200)))
    view_norm = norm01_log(views or 0, int(N.get("views_max", 100000)))
    return float(W.get("keyword", 3)) * kw_norm + float(W.get("upvotes", 5)) * up_norm + float(W.get("views", 2)) * view_norm

