# tools/url_check.py
from typing import Dict
import urllib.request
import urllib.error

def check_url(url: str, timeout_sec: float = 3.0, max_hops: int = 3) -> Dict:
    """
    Try HEAD first, then fallback to GET. Follow redirects up to max_hops.
    Returns: {
      "status": "ok" | "redirect" | "dead",
      "http_code": int|None,
      "final_url": str|None,
      "hops": int
    }
    """
    def _one(method: str, u: str):
        req = urllib.request.Request(u, method=method)
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            code = resp.getcode()
            final_url = resp.geturl()
            return code, final_url, resp.headers

    hops = 0
    cur = url
    last_code = None
    try:
        # try HEAD
        try:
            last_code, cur, headers = _one("HEAD", cur)
        except Exception:
            # fallback GET
            last_code, cur, headers = _one("GET", cur)

        # manual redirect loop (for clarity)
        while last_code and 300 <= int(last_code) < 400 and hops < max_hops:
            loc = headers.get("Location") or headers.get("location")
            if not loc:
                break
            hops += 1
            if loc.startswith("/"):  # relative -> resolve with current
                from urllib.parse import urljoin
                cur = urljoin(cur, loc)
            else:
                cur = loc
            # try HEAD again on new location
            try:
                last_code, cur, headers = _one("HEAD", cur)
            except Exception:
                last_code, cur, headers = _one("GET", cur)

        status = "dead"
        if last_code:
            if 200 <= int(last_code) < 300:
                status = "ok" if hops == 0 else "redirect"
            elif 300 <= int(last_code) < 400:
                status = "redirect"  # exceeded max_hops but still redirect
            else:
                status = "dead"

        return {"status": status, "http_code": last_code, "final_url": cur, "hops": hops}
    except urllib.error.URLError:
        return {"status": "dead", "http_code": None, "final_url": None, "hops": hops}
