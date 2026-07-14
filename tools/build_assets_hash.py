# -*- coding: utf-8 -*-
"""
tools/build_assets_hash.py
- 퀄리저널 레포 구조 자동 감지(admin/index.html 우선) 후, assets/*에 해시를 붙여 dist/ 생성
- HTML 내 assets 경로(<script src="assets/...">, <link href="assets/...">, <img src="assets/...">)를 해시 파일로 치환
- 결과물: <UI_BASE>/dist/index.html, <UI_BASE>/dist/assets/*.[hash].ext, <UI_BASE>/dist/manifest.json
사용법:
    python tools/build_assets_hash.py
옵션:
    --clean   : dist 폴더를 비우고 시작
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# UI 베이스 자동탐지: admin/index.html 있으면 admin, 없으면 레포 루트
UI_BASE = REPO / "admin" if (REPO / "admin" / "index.html").exists() else REPO
SRC_HTML = UI_BASE / "index.html"
SRC_DIR  = UI_BASE / "assets"
DIST     = UI_BASE / "dist"
DST_ASSETS = DIST / "assets"

def sha256_n(p: Path, n: int = 10) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:n]

def clean_dist():
    if DIST.exists():
        shutil.rmtree(DIST, ignore_errors=True)
    DIST.mkdir(parents=True, exist_ok=True)
    DST_ASSETS.mkdir(parents=True, exist_ok=True)

def build():
    if not SRC_HTML.exists():
        raise SystemExit(f"[ERR] index.html not found at: {SRC_HTML}")
    if not SRC_DIR.exists():
        SRC_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[WARN] {SRC_DIR} created (빈 assets 디렉터리).")

    manifest: dict[str, str] = {}
    # 1) assets -> dist/assets (해시 파일명으로 복사)
    for p in sorted(SRC_DIR.rglob("*"), key=lambda item: item.relative_to(SRC_DIR).as_posix()):
        if p.is_dir():
            continue
        rel = p.relative_to(SRC_DIR).as_posix()  # 예: js/app.js, style.css
        stem, ext = os.path.splitext(rel)
        h = sha256_n(p)
        hashed = f"{stem}.{h}{ext}"
        dst = DST_ASSETS / hashed
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Copy content only. Source mtimes and filesystem enumeration order must
        # not influence the generated artifact.
        shutil.copyfile(p, dst)
        manifest[f"assets/{rel}"] = f"assets/{hashed}"

    # 2) index.html 읽어서 assets 참조 치환
    html = SRC_HTML.read_text(encoding="utf-8")

    def rep_attr(pat: str, html: str) -> str:
        def replace_ref(m):
            url = m.group(2)
            # assets/ 경로만 해시치환 (절대경로, 외부 URL 등은 그대로)
            key = url.replace("\\", "/").lstrip("./")
            if not key.startswith("assets/"):
                return m.group(0)
            return f'{m.group(1)}{manifest.get(key, url)}{m.group(3)}'
        return re.sub(pat, replace_ref, html, flags=re.I)

    html = rep_attr(r'(<link[^>]+href=["\'])([^"\']+)(["\'])', html)
    html = rep_attr(r'(<script[^>]+src=["\'])([^"\']+)(["\'])', html)
    html = rep_attr(r'(<img[^>]+src=["\'])([^"\']+)(["\'])', html)

    # 3) dist/index.html + manifest.json 기록
    DIST.mkdir(parents=True, exist_ok=True)
    # Write explicit UTF-8 bytes so Windows/Linux newline translation cannot
    # make the same tracked source produce different dist content.
    (DIST / "index.html").write_bytes(html.encode("utf-8"))
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (DIST / "manifest.json").write_bytes(manifest_text.encode("utf-8"))

    print(f"[OK] 해시 적용 {len(manifest)}개 → {DIST} 생성 완료")
    print(f"     UI_BASE = {UI_BASE}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="dist 폴더를 비우고 시작")
    args = ap.parse_args()

    if args.clean:
        clean_dist()
    else:
        DIST.mkdir(parents=True, exist_ok=True)
        DST_ASSETS.mkdir(parents=True, exist_ok=True)

    build()

if __name__ == "__main__":
    main()
