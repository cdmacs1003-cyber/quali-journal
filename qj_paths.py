from qj_paths import rel as qj_rel
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
def rel(*parts) -> str:
    return str(ROOT_DIR.joinpath(*parts))

