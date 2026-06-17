from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ADMIN_SERVER_QUALI = REPO_ROOT / "admin" / "server_quali.py"
ROOT_SERVER_QUALI = REPO_ROOT / "server_quali.py"
APPROVED_LOGS_DIR = Path("/tmp/logs").resolve()
APPROVED_ARCHIVE_DIR = Path("/tmp/archive").resolve()
APPROVED_REPORT_DIR = (Path("/tmp/archive") / "reports").resolve()

MODULE_FILE_CONFIRMED = False
ROOT_SERVER_QUALI_IMPORTED = False
DB_READY_FORCED_FALSE = False
RUN_PY_SENTINEL_INSTALLED = False
SYNC_SENTINEL_INSTALLED = False
RUN_PY_SENTINEL_UNEXPECTED_TRIGGERED = False
SYNC_SENTINEL_UNEXPECTED_TRIGGERED = False
TEMP_BOUNDARY_OK = False


def _resolved(path: object) -> Path:
    return Path(path).resolve()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _assert_outside_repository(path: Path) -> None:
    if path == REPO_ROOT or _is_relative_to(path, REPO_ROOT):
        raise AssertionError(f"pytest temp boundary points inside repository: {path}")


def _assert_exact_path(actual: object, expected: Path, label: str) -> None:
    actual_path = _resolved(actual)
    expected_path = expected.resolve()
    if actual_path != expected_path:
        raise AssertionError(f"{label} resolved to {actual_path}, expected {expected_path}")
    _assert_outside_repository(actual_path)


def _unexpected_run_py(*_args: object, **_kwargs: object) -> object:
    global RUN_PY_SENTINEL_UNEXPECTED_TRIGGERED
    RUN_PY_SENTINEL_UNEXPECTED_TRIGGERED = True
    raise RuntimeError("R9ZNW-33 blocked unexpected server_quali._run_py call")


def _unexpected_sync_after_save(*_args: object, **_kwargs: object) -> object:
    global SYNC_SENTINEL_UNEXPECTED_TRIGGERED
    SYNC_SENTINEL_UNEXPECTED_TRIGGERED = True
    raise RuntimeError("R9ZNW-33 blocked unexpected server_quali._sync_after_save call")


def _load_admin_server_quali() -> object:
    existing = sys.modules.get("server_quali")
    if existing is not None:
        existing_file = _resolved(getattr(existing, "__file__", ""))
        raise AssertionError(
            "server_quali was imported before R9ZNW-33 import-control setup: "
            f"{existing_file}"
        )

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    spec = importlib.util.spec_from_file_location("server_quali", ADMIN_SERVER_QUALI)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load pytest-controlled module: {ADMIN_SERVER_QUALI}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["server_quali"] = module
    spec.loader.exec_module(module)
    return module


def _configure_import_control() -> object:
    global MODULE_FILE_CONFIRMED
    global ROOT_SERVER_QUALI_IMPORTED
    global DB_READY_FORCED_FALSE
    global RUN_PY_SENTINEL_INSTALLED
    global SYNC_SENTINEL_INSTALLED
    global TEMP_BOUNDARY_OK

    os.environ["QUALIJOURNAL_SKIP_DOTENV"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["K_SERVICE"] = "r9znw33-synthetic-temp-boundary"
    os.environ["GCS_BUCKET"] = ""
    os.environ["ADMIN_TOKEN"] = "r9znw33-synthetic-admin-token"
    sys.dont_write_bytecode = True

    module = _load_admin_server_quali()

    module_file = _resolved(getattr(module, "__file__", ""))
    admin_file = ADMIN_SERVER_QUALI.resolve()
    root_file = ROOT_SERVER_QUALI.resolve()
    if module_file != admin_file:
        raise AssertionError(f"server_quali resolved to {module_file}, expected {admin_file}")
    if module_file == root_file:
        raise AssertionError("root server_quali.py was imported as server_quali")
    MODULE_FILE_CONFIRMED = True
    ROOT_SERVER_QUALI_IMPORTED = False

    setattr(module, "_DB_READY", False)
    if getattr(module, "_DB_READY", None) is not False:
        raise AssertionError("server_quali._DB_READY could not be forced false")
    DB_READY_FORCED_FALSE = True

    setattr(module, "_run_py", _unexpected_run_py)
    setattr(module, "_sync_after_save", _unexpected_sync_after_save)
    RUN_PY_SENTINEL_INSTALLED = True
    SYNC_SENTINEL_INSTALLED = True

    _assert_exact_path(getattr(module, "LOGS_DIR"), APPROVED_LOGS_DIR, "LOGS_DIR")
    _assert_exact_path(getattr(module, "ARCHIVE"), APPROVED_ARCHIVE_DIR, "ARCHIVE")
    _assert_exact_path(getattr(module, "REPORT_DIR"), APPROVED_REPORT_DIR, "REPORT_DIR")
    TEMP_BOUNDARY_OK = True
    return module


SERVER_QUALI = _configure_import_control()


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ARG001
    terminalreporter.write_line(f"R9ZNW33_MODULE_FILE={Path(SERVER_QUALI.__file__).resolve()}")
    terminalreporter.write_line(f"R9ZNW33_MODULE_FILE_CONFIRMED={MODULE_FILE_CONFIRMED}")
    terminalreporter.write_line(f"R9ZNW33_ROOT_SERVER_QUALI_IMPORTED={ROOT_SERVER_QUALI_IMPORTED}")
    terminalreporter.write_line(f"R9ZNW33_DB_READY_FORCED_FALSE={DB_READY_FORCED_FALSE}")
    terminalreporter.write_line(f"R9ZNW33_RUN_PY_SENTINEL_INSTALLED={RUN_PY_SENTINEL_INSTALLED}")
    terminalreporter.write_line(f"R9ZNW33_SYNC_SENTINEL_INSTALLED={SYNC_SENTINEL_INSTALLED}")
    terminalreporter.write_line(
        f"R9ZNW33_RUN_PY_SENTINEL_UNEXPECTED_TRIGGERED={RUN_PY_SENTINEL_UNEXPECTED_TRIGGERED}"
    )
    terminalreporter.write_line(
        f"R9ZNW33_SYNC_SENTINEL_UNEXPECTED_TRIGGERED={SYNC_SENTINEL_UNEXPECTED_TRIGGERED}"
    )
    terminalreporter.write_line(f"R9ZNW33_TEMP_BOUNDARY_OK={TEMP_BOUNDARY_OK}")
