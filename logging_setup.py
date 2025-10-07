from qj_paths import rel as qj_rel
# logging_setup.py — 최소 버전 (콘솔+파일 로깅)
import logging
from pathlib import Path

def setup_logger(name: str, logfile: str):
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 중복 추가 방지
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

