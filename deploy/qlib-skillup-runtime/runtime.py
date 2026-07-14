"""Minimal runtime entry point for the private QLIB Skillup field beta."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from admin.f13_bridge_api import router as f13_bridge_router


APP_ROOT = Path(__file__).resolve().parent
DIST_ROOT = APP_ROOT / "dist"
if not DIST_ROOT.is_dir():
    # Repository-local verification uses the canonical generated dist. The
    # container copies the same tree next to this entry point.
    DIST_ROOT = APP_ROOT.parents[1] / "admin" / "dist"

app = FastAPI(
    title="QLIB Skillup Runtime",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.include_router(f13_bridge_router)
app.mount("/assets", StaticFiles(directory=DIST_ROOT / "assets"), name="assets")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "qlib-skillup-runtime"}


@app.get("/", include_in_schema=False)
def beginner_ui() -> FileResponse:
    return FileResponse(DIST_ROOT / "index.html")
