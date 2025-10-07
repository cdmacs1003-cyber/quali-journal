from qj_paths import rel as qj_rel
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json, time, math, textwrap
from typing import List, Dict, Tuple, Optional
import urllib.request, urllib.error

# ------------------------------
# Sentence split (very simple)
# ------------------------------
_SENT_SPLIT = re.compile(r'(?<=[.!??귨펯竊?)\s+')
def split_sentences(txt: str) -> List[str]:
    txt = (txt or "").strip()
    if not txt:
        return []
    # normalize whitespace
    txt = re.sub(r'\s+', ' ', txt)
    sents = _SENT_SPLIT.split(txt)
    # fallback if no punctuation
    if len(sents) == 1 and len(txt) > 0:
        sents = re.split(r'(?<=\.)\s+|(?<=\!)\s+|(?<=\?)\s+', txt)
    # strip and dedup quasi-duplicates
    out, seen = [], set()
    for s in sents:
        s = s.strip()
        k = s.lower()
        if s and k not in seen:
            out.append(s)
            seen.add(k)
    return out

# ------------------------------
# Simple extractive summary
# ------------------------------
_STOPWORDS = set('a an the of to in on and or for from with by as is are was were be been being this that these those it its their his her our your my we they i you he she them there here than then also into over under out up down off across about after before during among within without between per via using use used new more most first last'.split())

def keyword_score(text: str, title_hint: str = "") -> Dict[str, int]:
    words = re.findall(r'[A-Za-z0-9\-]+', (text or "").lower())
    hint = set(re.findall(r'[A-Za-z0-9\-]+', (title_hint or "").lower()))
    scores = {}
    for w in words:
        if w in _STOPWORDS or len(w) <= 2:
            continue
        scores[w] = scores.get(w, 0) + 1
    for w in hint:
        if w in _STOPWORDS or len(w) <= 2:
            continue
        scores[w] = scores.get(w, 0) + 2  # boost title terms
    return scores

def sentence_score(sent: str, tf: Dict[str,int]) -> float:
    toks = re.findall(r'[A-Za-z0-9\-]+', (sent or "").lower())
    if not toks:
        return 0.0
    score = 0.0
    for t in toks:
        if t in _STOPWORDS or len(t) <= 2:
            continue
        score += tf.get(t, 0)
    # prefer shorter sentences slightly
    score = score / max(1.0, math.log(len(sent)+1, 1.7))
    return score

def extractive_summary(text: str, title_hint: str = "", max_sent: int = 3) -> str:
    sents = split_sentences(text)
    if not sents:
        return ""
    tf = keyword_score(text, title_hint)
    ranked = sorted(sents, key=lambda s: sentence_score(s, tf), reverse=True)
    top = ranked[:max_sent]
    # keep the original order for readability
    ordered = [s for s in sents if s in top]
    return " ".join(ordered)

# ------------------------------
# Optional OpenAI call (translation / refine summary)
# ------------------------------
def _openai_chat(messages, model: str = None, api_key: Optional[str] = None, timeout: int = 30) -> Optional[str]:
    api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        return None
    model = model or os.getenv("OPENAI_TRANSLATE_MODEL") or "gpt-4o-mini"
    import json as _json
    req = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 400
    }
    data = _json.dumps(req).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    try:
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
            obj = _json.loads(payload)
            return obj["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

def to_ko(text_en: str) -> str:
    if not text_en:
        return ""
    out = _openai_chat([
        {"role":"system","content":"You are a professional Korean technical editor. Translate concisely into natural Korean."},
        {"role":"user","content":f"Translate into Korean, keep it within 2-3 short sentences:\n\n{text_en}"}
    ])
    return out or text_en  # fallback: return English if API not available

# ------------------------------
# Core enrich logic
# ------------------------------
def enrich_item(it: dict, max_sent: int = 3) -> dict:
    """Return a copy with summary_en/summary_ko added if missing."""
    t = (it.get("title") or "").strip()
    desc = (it.get("desc") or it.get("summary") or it.get("content") or "").strip()
    if not desc:
        # Try from body if present
        desc = (it.get("body") or "").strip()
    out = dict(it)
    if not out.get("summary_en"):
        out["summary_en"] = extractive_summary(desc, t, max_sent=max_sent) or (desc[:300] + ("..." if len(desc)>300 else ""))
    if not out.get("summary_ko"):
        out["summary_ko"] = to_ko(out["summary_en"])
    return out

def detect_article_list(j: dict) -> Tuple[str, list]:
    # Try common layouts
    if "articles" in j and isinstance(j["articles"], list):
        return "articles", j["articles"]
    if "news" in j and isinstance(j["news"], list):
        return "news", j["news"]
    sec = j.get("sections",{})
    for k in ("news","articles","?쒖? ?댁뒪","湲濡쒕쾶 ?꾩옄?앹궛","AI ?댁뒪","而ㅻ??덊떚"):
        if k in sec and isinstance(sec[k], list):
            return f"sections.{k}", sec[k]
    # default
    return "articles", j.get("articles", [])

def set_article_list(j: dict, keypath: str, arr: list):
    if keypath == "articles":
        j["articles"] = arr; return
    if keypath == "news":
        j["news"] = arr; return
    if keypath.startswith("sections."):
        sec_key = keypath.split(".",1)[1]
        if "sections" not in j or not isinstance(j["sections"], dict):
            j["sections"] = {}
        j["sections"][sec_key] = arr
        return
    # default
    j["articles"] = arr

