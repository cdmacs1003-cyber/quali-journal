# qj_paths.py
# 프로젝트 루트 기준 경로 유틸 (멱등/순환임포트 없음)
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Union

# 프로젝트 루트: 이 파일이 놓인 폴더
ROOT_DIR: Path = Path(__file__).resolve().parent

def rel(*parts: Union[str, Path]) -> str:
    """
    프로젝트 루트(ROOT_DIR)를 기준으로 하위 경로를 만들어 절대경로(str)로 반환.
    예) rel("data", "selected_articles.json")
    """
    return str(ROOT_DIR.joinpath(*parts))

def ensure_parent(path: Union[str, Path]) -> str:
    """
    파일 경로의 상위 폴더를 모두 생성(parents=True)한 뒤, 동일 경로(str)를 반환.
    쓰기 전에 폴더 보장할 때 사용.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)

__all__ = ["ROOT_DIR", "rel", "ensure_parent"]

if __name__ == "__main__":
    # 간단 자가테스트
    print("ROOT_DIR =", ROOT_DIR)
    print("sample   =", rel("data", "sample.txt"))
