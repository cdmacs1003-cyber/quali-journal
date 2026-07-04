# -*- coding: utf-8 -*-
"""QLIB Warehouse core API.

This module implements the warehouse contract from the project guidebook:
manifest, item quarantine, raw hash preservation, provenance/rights gates,
review/approval, promotion dry-run, promotion trace, backup/restore dry-run,
release board, and validator output.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/warehouse", tags=["warehouse"])

ITEM_TYPES = {
    "tacit_knowledge",
    "expert_knowhow",
    "personal_paper",
    "report",
    "reporter_note",
    "contribution",
    "community_contribution",
    "failure_record",
    "review_memo",
    "standard_note",
    "field_case",
    "education_seed",
    "analytics_improvement_candidate",
    "raw_document",
}

STATUSES = {
    "captured",
    "untriaged",
    "triaged",
    "needs_source",
    "pending_review",
    "private_tacit",
    "hold",
    "duplicate_suspected",
    "rejected",
    "archived",
    "approved_for_library",
    "promoted",
}

STATUS_TRANSITIONS = {
    "captured": {"untriaged"},
    "untriaged": {"triaged"},
    "triaged": {"needs_source", "pending_review", "private_tacit", "duplicate_suspected", "rejected"},
    "needs_source": {"pending_review"},
    "pending_review": {"hold", "rejected", "approved_for_library"},
    "approved_for_library": {"promoted"},
    "hold": {"pending_review", "archived"},
    "duplicate_suspected": {"archived", "pending_review"},
    "rejected": {"archived"},
}

RIGHTS_STATUSES = {
    "owned",
    "licensed",
    "permission_granted",
    "public_reference",
    "internal_only",
    "no_export",
    "unknown",
}

SENSITIVITIES = {"public", "internal", "restricted", "private", "secret"}

VISIBILITIES = {
    "warehouse_only",
    "reviewer_only",
    "library_candidate",
    "library_internal",
    "public_summary_allowed",
    "no_export",
}

PROMOTION_TARGETS = {
    "library_standard_card",
    "library_reference_card",
    "library_case_card",
    "library_training_seed",
    "library_failure_lesson",
    "library_tailoring_note",
    "library_graph_node_only",
}

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?i)(api[_\- ]?key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{10,}"),
    re.compile(r"(?i)-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


class Provenance(BaseModel):
    source_type: str = Field(default="unknown")
    source_title: str
    source_author: str = Field(default="")
    source_org: str = Field(default="")
    source_date: str = Field(default="")
    captured_by: str
    captured_at: Optional[str] = None
    source_locator: str = Field(default="")


class WarehouseCreateItemReq(BaseModel):
    item_type: Literal[
        "tacit_knowledge",
        "expert_knowhow",
        "personal_paper",
        "report",
        "reporter_note",
        "contribution",
        "community_contribution",
        "failure_record",
        "review_memo",
        "standard_note",
        "field_case",
        "education_seed",
        "analytics_improvement_candidate",
        "raw_document",
    ]
    title: str
    summary: str = Field(default="")
    raw_text: Optional[str] = None
    raw_pointer: Optional[str] = None
    raw_mime_type: str = Field(default="text/plain")
    provenance: Provenance
    rights_status: Literal[
        "owned",
        "licensed",
        "permission_granted",
        "public_reference",
        "internal_only",
        "no_export",
        "unknown",
    ] = "unknown"
    sensitivity: Literal["public", "internal", "restricted", "private", "secret"] = "internal"
    visibility: Literal[
        "warehouse_only",
        "reviewer_only",
        "library_candidate",
        "library_internal",
        "public_summary_allowed",
        "no_export",
    ] = "warehouse_only"
    tags: List[str] = Field(default_factory=list)


class WarehouseStatusReq(BaseModel):
    status: Literal[
        "captured",
        "untriaged",
        "triaged",
        "needs_source",
        "pending_review",
        "private_tacit",
        "hold",
        "duplicate_suspected",
        "rejected",
        "archived",
        "approved_for_library",
        "promoted",
    ]
    reason: str = Field(default="")
    actor_id: str = Field(default="system")


class WarehouseReviewReq(BaseModel):
    reviewer_id: str
    reviewer_role: str = Field(default="Subject Reviewer")
    review_decision: Literal["needs_source", "pending_review", "hold", "rejected", "approved_for_library"]
    review_note: str
    quality_score: int = Field(ge=0, le=100)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    rights_status_confirmed: bool = False
    sensitivity_confirmed: bool = False
    promotion_recommendation: Literal[
        "library_standard_card",
        "library_reference_card",
        "library_case_card",
        "library_training_seed",
        "library_failure_lesson",
        "library_tailoring_note",
        "library_graph_node_only",
        "none",
    ] = "none"


class WarehouseApproveReq(BaseModel):
    approver_id: str
    approval_note: str


class LibraryBinding(BaseModel):
    org: str = Field(default="QLIB")
    doc_code: Optional[str] = None
    rev: str = Field(default="v1")
    year: Optional[int] = None
    source_lang: str = Field(default="EN")
    doc_kind: Literal["STANDARD", "REFERENCE"] = "REFERENCE"
    title_en: Optional[str] = None
    search_query: Optional[str] = None


class PromotionDryRunReq(BaseModel):
    promotion_target: Literal[
        "library_standard_card",
        "library_reference_card",
        "library_case_card",
        "library_training_seed",
        "library_failure_lesson",
        "library_tailoring_note",
        "library_graph_node_only",
    ] = "library_reference_card"
    created_by: str = Field(default="system")
    library_binding: Optional[LibraryBinding] = None


class PromoteReq(PromotionDryRunReq):
    promotion_dry_run_id: Optional[str] = None


class BackupRunReq(BaseModel):
    created_by: str = Field(default="system")
    dry_run_only: bool = False


class ReleaseBoardReq(BaseModel):
    scope: str = Field(default="warehouse")
    changed_files: List[str] = Field(default_factory=list)
    test_results: List[Dict[str, Any]] = Field(default_factory=list)
    rollback_plan: str = Field(default="restore latest warehouse backup")
    approver: str = Field(default="system")
    handover_path: str = Field(default="")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _project_root() -> Path:
    env = os.getenv("QUALI_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    if os.getenv("K_SERVICE"):
        return Path("/tmp").resolve()
    return Path(__file__).resolve().parent.parent


def _paths() -> Dict[str, Path]:
    project = _project_root()
    warehouse_root = Path(os.getenv("QUALI_WAREHOUSE_ROOT") or (project / "data" / "warehouse")).resolve()
    library_root = Path(os.getenv("QUALI_LIBRARY_ROOT") or (project / "data" / "library")).resolve()
    return {
        "project_root": project,
        "warehouse_root": warehouse_root,
        "raw_root": warehouse_root / "raw",
        "derived_root": warehouse_root / "derived",
        "summary_root": warehouse_root / "derived" / "summary",
        "validation_root": warehouse_root / "derived" / "validation",
        "trace_root": warehouse_root / "trace",
        "dry_run_root": warehouse_root / "trace" / "dry_run",
        "schema_root": warehouse_root / "schema",
        "review_root": warehouse_root / "review",
        "indexes_root": warehouse_root / "indexes",
        "library_root": library_root,
        "library_card_root": library_root / "cards" / "reference",
        "library_graph_root": library_root / "graph",
        "library_evidence_root": library_root / "evidence",
        "backup_root": Path(os.getenv("QUALI_WAREHOUSE_BACKUP_ROOT") or (project / "backup" / "warehouse")).resolve(),
        "proofpack_root": Path(os.getenv("QUALI_WAREHOUSE_PROOFPACK_ROOT") or (project / "reports" / "proofpacks" / "warehouse")).resolve(),
        "release_root": Path(os.getenv("QUALI_WAREHOUSE_RELEASE_ROOT") or (project / "releases" / "warehouse")).resolve(),
    }


def _ensure_layout() -> Dict[str, Path]:
    paths = _paths()
    for key, path in paths.items():
        if key.endswith("_root") or key in {"warehouse_root", "backup_root"}:
            path.mkdir(parents=True, exist_ok=True)
    _write_static_schema_files(paths)
    _write_manifest(paths)
    return paths


def _manifest_dict(paths: Dict[str, Path]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "module_id": "QLIB-WAREHOUSE",
        "role": "library_pre_approval_warehouse",
        "owner_project": "qlib",
        "date": "2026-05-14",
        "roots": {
            "warehouse_root": str(paths["warehouse_root"]),
            "raw_root": str(paths["raw_root"]),
            "derived_root": str(paths["derived_root"]),
            "trace_root": str(paths["trace_root"]),
            "schema_root": str(paths["schema_root"]),
            "review_root": str(paths["review_root"]),
            "backup_root": str(paths["backup_root"]),
            "proofpack_root": str(paths["proofpack_root"]),
            "release_root": str(paths["release_root"]),
        },
        "official_indexes": {
            "warehouse_items": str(paths["warehouse_root"] / "warehouse_items.jsonl"),
            "promotion_trace": str(paths["trace_root"] / "promotion_trace.jsonl"),
        },
        "rules": {
            "raw_is_immutable": True,
            "every_item_requires_source_id": True,
            "every_item_requires_provenance": True,
            "every_item_requires_raw_hash": True,
            "approved_items_require_review": True,
            "promotion_requires_dry_run": True,
            "library_write_from_warehouse_ui": False,
            "backup_restore_test_required": True,
        },
    }


def _write_manifest(paths: Dict[str, Path]) -> None:
    _write_json(paths["warehouse_root"] / "warehouse_manifest.json", _manifest_dict(paths))


def _write_static_schema_files(paths: Dict[str, Path]) -> None:
    item_schema = {
        "schema_version": 1,
        "required": [
            "warehouse_item_id",
            "source_id",
            "item_type",
            "status",
            "title",
            "summary",
            "raw_text_ref",
            "raw_hash",
            "provenance",
            "rights_status",
            "sensitivity",
            "visibility",
            "quality_score",
            "confidence",
        ],
        "enums": {
            "item_type": sorted(ITEM_TYPES),
            "status": sorted(STATUSES),
            "rights_status": sorted(RIGHTS_STATUSES),
            "sensitivity": sorted(SENSITIVITIES),
            "visibility": sorted(VISIBILITIES),
        },
    }
    trace_schema = {
        "schema_version": 1,
        "required": [
            "promotion_trace_id",
            "warehouse_item_id",
            "warehouse_item_hash",
            "raw_hash",
            "source_item_status",
            "promotion_target",
            "validation_result",
            "policy_result",
            "output_artifacts",
            "created_by",
            "created_at",
        ],
        "promotion_targets": sorted(PROMOTION_TARGETS),
    }
    _write_json(paths["schema_root"] / "warehouse_item_schema.json", item_schema)
    _write_json(paths["schema_root"] / "promotion_trace_schema.json", trace_schema)


def _dump_model(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == payload:
                return
        except Exception:
            pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    try:
        tmp.replace(path)
    except PermissionError:
        if path.exists():
            try:
                if path.read_text(encoding="utf-8") == payload:
                    tmp.unlink(missing_ok=True)
                    return
            except Exception:
                pass
        raise


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                rows.append(data)
        except Exception:
            continue
    return rows


def _items_path() -> Path:
    return _paths()["warehouse_root"] / "warehouse_items.jsonl"


def _trace_path() -> Path:
    return _paths()["trace_root"] / "promotion_trace.jsonl"


def _latest_items() -> Dict[str, Dict[str, Any]]:
    rows = _read_jsonl(_items_path())
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        wid = str(row.get("warehouse_item_id") or "")
        if not wid:
            continue
        prev = latest.get(wid)
        if not prev or int(row.get("revision") or 0) >= int(prev.get("revision") or 0):
            latest[wid] = row
    return latest


def _get_item(item_id: str) -> Dict[str, Any]:
    item = _latest_items().get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail={"code": "WH-404", "message": "warehouse item not found"})
    return item


def _save_item(item: Dict[str, Any], *, new_revision: bool = True) -> Dict[str, Any]:
    if new_revision:
        item = dict(item)
        item["revision"] = int(item.get("revision") or 0) + 1
        item["updated_at"] = _now()
    _append_jsonl(_items_path(), item)
    return item


def _hash_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _hash_text(text: str) -> str:
    return _hash_bytes(text.encode("utf-8"))


def _short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:n].upper()


def _id(prefix: str, seed: str) -> str:
    return f"{prefix}-{_stamp()}-{_short_hash(seed, 4)}"


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path)


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _library_engine_enabled() -> bool:
    return _truthy_env("QUALI_LIBRARY_RIPPLE_ENABLED")


def _safe_token(value: Any, fallback: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-").lower()
    return (text or fallback.lower()).strip("-")[:80]


def _source_year(item: Dict[str, Any]) -> int:
    candidates = [
        ((item.get("provenance") or {}).get("source_date") if isinstance(item.get("provenance"), dict) else ""),
        item.get("created_at"),
    ]
    for value in candidates:
        match = re.search(r"(?:19|20)\d{2}", str(value or ""))
        if match:
            return int(match.group(0))
    return datetime.now(timezone.utc).year


def _library_ltm_root(paths: Optional[Dict[str, Path]] = None) -> Path:
    explicit = os.getenv("QUALI_LIBRARY_LTM_ROOT") or os.getenv("LTM_ROOT")
    if explicit:
        return Path(explicit).resolve()
    resolved_paths = paths or _paths()
    return resolved_paths["library_root"].parent.resolve()


def _raw_path_for_item(item: Dict[str, Any]) -> Path:
    raw_ref = str(item.get("raw_text_ref") or "")
    raw_path = Path(raw_ref)
    if raw_path.is_absolute():
        return raw_path
    return (_paths()["warehouse_root"] / raw_ref).resolve()


def _default_search_query(item: Dict[str, Any]) -> str:
    raw_path = _raw_path_for_item(item)
    candidates: List[str] = []
    if raw_path.exists():
        try:
            raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")
            words = re.findall(r"[A-Za-z0-9가-힣_-]{3,}", raw_text)
            if words:
                candidates.append(" ".join(words[:4]))
        except Exception:
            pass
    candidates.extend([str(item.get("summary") or ""), str(item.get("title") or "")])
    for candidate in candidates:
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if candidate:
            return candidate[:120]
    return str(item.get("warehouse_item_id") or "warehouse")


def _library_binding_for_item(item: Dict[str, Any], req: PromotionDryRunReq, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if isinstance(override, dict):
        data.update({k: v for k, v in override.items() if v not in {None, ""}})
    elif req.library_binding is not None:
        data.update({k: v for k, v in _dump_model(req.library_binding).items() if v not in {None, ""}})

    default_kind = "STANDARD" if req.promotion_target == "library_standard_card" else "REFERENCE"
    org = _safe_token(data.get("org") or "QLIB", "QLIB").upper()
    doc_code = _safe_token(data.get("doc_code") or item.get("title") or item.get("warehouse_item_id"), str(item.get("warehouse_item_id") or "warehouse-item"))
    rev = str(data.get("rev") or "v1").strip() or "v1"
    year = int(data.get("year") or _source_year(item))
    source_lang = str(data.get("source_lang") or "EN").strip().upper() or "EN"
    doc_kind = str(data.get("doc_kind") or default_kind).strip().upper()
    if doc_kind not in {"STANDARD", "REFERENCE"}:
        doc_kind = default_kind
    title_en = str(data.get("title_en") or item.get("title") or doc_code).strip()
    search_query = str(data.get("search_query") or _default_search_query(item) or doc_code).strip()
    return {
        "org": org,
        "doc_code": doc_code,
        "rev": rev,
        "year": year,
        "source_lang": source_lang,
        "doc_kind": doc_kind,
        "title_en": title_en,
        "search_query": search_query,
        "node_id": f"{org}:{doc_code}@{rev}",
    }


def _library_expected_artifacts(item: Dict[str, Any], binding: Dict[str, Any], ltm_root: Path) -> Dict[str, str]:
    suffix = _raw_path_for_item(item).suffix or ".txt"
    canonical_lang = "EN"
    base = f"{binding['org']}__{binding['doc_code']}__{binding['rev']}__{binding['year']}__{canonical_lang}__{binding['doc_kind']}__v1"
    card_dir = "standard_cards" if binding["doc_kind"] == "STANDARD" else "reference_cards"
    card_name = f"{binding['org']}__{binding['doc_code']}__{binding['rev']}.md"
    return {
        "ltm_root": str(ltm_root),
        "brain_db_path": str(ltm_root / "brain.db"),
        "graph_db_path": str(ltm_root / "graph.db"),
        "library_raw_path": str(ltm_root / "LIBRARY" / "raw" / f"{base}{suffix}"),
        "library_template_path": str(ltm_root / "LIBRARY" / "templates" / f"library_item.{base}{suffix}.yml"),
        "library_card_path": str(ltm_root / "LIBRARY" / "exports" / card_dir / card_name),
        "ripple_index_path": str(ltm_root / "LIBRARY" / "ripple" / "ripple_index.sqlite"),
    }


def _ripple_env(ltm_root: Path) -> Dict[str, str]:
    env = dict(os.environ)
    env["LTM_ROOT"] = str(ltm_root)
    extra_path = os.getenv("QUALI_LIBRARY_RIPPLE_PYTHONPATH")
    if extra_path:
        previous = env.get("PYTHONPATH")
        env["PYTHONPATH"] = extra_path if not previous else extra_path + os.pathsep + previous
    return env


def _ripple_timeout() -> int:
    try:
        return max(10, int(os.getenv("QUALI_LIBRARY_RIPPLE_TIMEOUT") or "120"))
    except Exception:
        return 120


def _run_ripple(args: List[str], ltm_root: Path) -> Dict[str, Any]:
    python_bin = os.getenv("QUALI_LIBRARY_RIPPLE_PYTHON") or sys.executable
    command = [python_bin, "-m", "qualilibrary_ripple"] + [str(arg) for arg in args]
    cwd = os.getenv("QUALI_LIBRARY_RIPPLE_CWD") or str(_project_root())
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=_ripple_env(ltm_root),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=_ripple_timeout(),
            check=False,
        )
        return {
            "command": subprocess.list2cmdline(command),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "ok": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": subprocess.list2cmdline(command),
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "\nTIMEOUT",
            "ok": False,
        }
    except Exception as exc:
        return {
            "command": subprocess.list2cmdline(command),
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "ok": False,
        }


def _ripple_add_args(raw_path: Path, binding: Dict[str, Any], dry_run: bool) -> List[str]:
    args = [
        "add",
        str(raw_path),
        binding["org"],
        binding["doc_code"],
        binding["rev"],
        str(binding["year"]),
        "--source-lang",
        binding["source_lang"],
        "--doc-kind",
        binding["doc_kind"],
        "--title-en",
        binding["title_en"],
    ]
    if dry_run:
        args.extend(["--dry-run", "--no-db", "--no-map", "--no-ripple"])
    return args


def _parse_add_result(result: Dict[str, Any], binding: Dict[str, Any]) -> Dict[str, Any]:
    text = str(result.get("stdout") or "") + "\n" + str(result.get("stderr") or "")
    memory_match = re.search(r"memory_item_id\s*=\s*(\d+)", text)
    node_match = re.search(r"(?:reference_node|standard_node)\s*=\s*([A-Za-z0-9_:\-.@]+)", text)
    return {
        "memory_item_id": int(memory_match.group(1)) if memory_match else None,
        "node_id": node_match.group(1) if node_match else binding["node_id"],
    }


def _parse_search_hits(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    stdout = str(result.get("stdout") or "").strip()
    try:
        parsed = json.loads(stdout)
    except Exception:
        match = re.search(r"(\[\s*\{.*\}\s*\])", stdout, flags=re.S)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(1))
        except Exception:
            return []
    return parsed if isinstance(parsed, list) else []


def _library_engine_disabled_result() -> Dict[str, Any]:
    return {
        "enabled": False,
        "adapter": "qualilibrary_ripple",
        "decision": "NOT_EXECUTED",
        "reason": "Set QUALI_LIBRARY_RIPPLE_ENABLED=1 to run real QualiLibrary Ripple promotion.",
    }


def _library_engine_dry_run(item: Dict[str, Any], req: PromotionDryRunReq) -> Dict[str, Any]:
    if not _library_engine_enabled():
        return _library_engine_disabled_result()
    ltm_root = _library_ltm_root()
    binding = _library_binding_for_item(item, req)
    raw_path = _raw_path_for_item(item)
    expected = _library_expected_artifacts(item, binding, ltm_root)
    if not raw_path.exists():
        return {
            "enabled": True,
            "adapter": "qualilibrary_ripple",
            "decision": "HOLD",
            "binding": binding,
            "ltm_root": str(ltm_root),
            "expected_artifacts": expected,
            "issues": [_issue("RAW_FILE_MISSING", "BLOCKER", "warehouse raw file is missing")],
        }
    result = _run_ripple(_ripple_add_args(raw_path, binding, dry_run=True), ltm_root)
    decision = "PASS" if result["ok"] else "HOLD"
    issues = [] if result["ok"] else [_issue("QUALILIBRARY_DRY_RUN_FAILED", "BLOCKER", "qualilibrary_ripple add --dry-run failed")]
    return {
        "enabled": True,
        "adapter": "qualilibrary_ripple",
        "decision": decision,
        "binding": binding,
        "ltm_root": str(ltm_root),
        "expected_artifacts": expected,
        "dry_run_result": result,
        "issues": issues,
    }


def _library_engine_promote(item: Dict[str, Any], req: PromoteReq, dry_run: Dict[str, Any]) -> Dict[str, Any]:
    if not _library_engine_enabled():
        return _library_engine_disabled_result()

    dry_engine = dry_run.get("library_engine") if isinstance(dry_run.get("library_engine"), dict) else {}
    if not dry_engine.get("enabled") or dry_engine.get("decision") != "PASS":
        raise HTTPException(
            status_code=409,
            detail={"code": "WH-LIBRARY-DRYRUN-MISSING", "message": "real QualiLibrary Ripple dry-run must PASS before promotion"},
        )

    binding_override = dry_engine.get("binding") if req.library_binding is None else None
    binding = _library_binding_for_item(item, req, binding_override if isinstance(binding_override, dict) else None)
    ltm_root = _library_ltm_root()
    raw_path = _raw_path_for_item(item)
    expected = _library_expected_artifacts(item, binding, ltm_root)
    if not raw_path.exists():
        raise HTTPException(status_code=409, detail={"code": "WH-RAW-MISSING", "message": "warehouse raw file is missing"})

    add_result = _run_ripple(_ripple_add_args(raw_path, binding, dry_run=False), ltm_root)
    add_meta = _parse_add_result(add_result, binding)
    verify_result = _run_ripple(["verify"], ltm_root) if add_result["ok"] else {"ok": False, "skipped": True, "reason": "add failed"}
    rebuild_result = _run_ripple(["ripple", "rebuild"], ltm_root) if verify_result.get("ok") else {"ok": False, "skipped": True, "reason": "verify failed"}
    search_attempts: List[Dict[str, Any]] = []
    search_result: Dict[str, Any] = {"ok": False, "skipped": True, "reason": "rebuild failed", "stdout": "[]"}
    search_hits: List[Dict[str, Any]] = []
    if rebuild_result.get("ok"):
        query_candidates = [binding["search_query"], _default_search_query(item), str(item.get("summary") or ""), str(item.get("title") or "")]
        seen_queries = set()
        for query in query_candidates:
            query = re.sub(r"\s+", " ", str(query or "")).strip()
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)
            attempt = _run_ripple(["ripple", "search", query, "--k", "5"], ltm_root)
            attempt["query"] = query
            search_attempts.append(attempt)
            hits = _parse_search_hits(attempt)
            if attempt.get("ok") and hits:
                search_result = attempt
                search_hits = hits
                break
        if not search_attempts:
            search_result = {"ok": False, "skipped": True, "reason": "no search query candidates", "stdout": "[]"}
    show_result = _run_ripple(["show", add_meta["node_id"]], ltm_root) if search_result.get("ok") else {"ok": False, "skipped": True, "reason": "search failed"}
    expected_exists = {name: Path(path).exists() for name, path in expected.items() if name.endswith("_path")}
    required_paths = ["brain_db_path", "graph_db_path", "library_raw_path", "library_template_path", "library_card_path", "ripple_index_path"]
    required_paths_ok = all(expected_exists.get(name) for name in required_paths)
    decision = "PASS" if add_result["ok"] and verify_result.get("ok") and rebuild_result.get("ok") and search_result.get("ok") and search_hits and required_paths_ok else "HOLD"
    issues: List[Dict[str, str]] = []
    if not add_result["ok"]:
        issues.append(_issue("QUALILIBRARY_ADD_FAILED", "BLOCKER", "qualilibrary_ripple add failed"))
    if not verify_result.get("ok"):
        issues.append(_issue("QUALILIBRARY_VERIFY_FAILED", "BLOCKER", "qualilibrary_ripple verify failed"))
    if not rebuild_result.get("ok"):
        issues.append(_issue("QUALILIBRARY_RIPPLE_REBUILD_FAILED", "BLOCKER", "qualilibrary_ripple ripple rebuild failed"))
    if not search_result.get("ok") or not search_hits:
        issues.append(_issue("QUALILIBRARY_RIPPLE_SEARCH_FAILED", "BLOCKER", "qualilibrary_ripple ripple search did not return a verified hit"))
    if not required_paths_ok:
        issues.append(_issue("QUALILIBRARY_ARTIFACT_MISSING", "BLOCKER", "one or more required QualiLibrary artifacts are missing"))

    result = {
        "enabled": True,
        "adapter": "qualilibrary_ripple",
        "decision": decision,
        "binding": binding,
        "ltm_root": str(ltm_root),
        "expected_artifacts": expected,
        "artifact_exists": expected_exists,
        "memory_item_id": add_meta["memory_item_id"],
        "node_id": add_meta["node_id"],
        "add_result": add_result,
        "verify_result": verify_result,
        "ripple_rebuild_result": rebuild_result,
        "ripple_search_result": search_result,
        "ripple_search_attempts": search_attempts,
        "ripple_search_hits": search_hits,
        "show_result": show_result,
        "issues": issues,
    }
    if decision != "PASS":
        raise HTTPException(
            status_code=409,
            detail={"code": "WH-LIBRARY-PROMOTION-FAILED", "message": "real QualiLibrary Ripple promotion failed", "library_engine": result},
        )
    return result


def _raw_payload(req: WarehouseCreateItemReq) -> bytes:
    if req.raw_text is not None and req.raw_text != "":
        return req.raw_text.encode("utf-8")
    pointer = req.raw_pointer or req.provenance.source_locator or req.provenance.source_title
    if not pointer:
        raise HTTPException(status_code=422, detail={"code": "WH-RAW-001", "message": "raw_text or raw_pointer is required"})
    return ("RAW_POINTER\n" + pointer).encode("utf-8")


def _write_raw_file(paths: Dict[str, Path], item_id: str, data: bytes, mime_type: str) -> Dict[str, Any]:
    ym = datetime.now(timezone.utc).strftime("%Y/%m")
    suffix = ".json" if mime_type == "application/json" else ".txt"
    raw_path = paths["raw_root"] / ym / f"{item_id}{suffix}"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    digest = _hash_bytes(data)
    if raw_path.exists():
        existing = _hash_bytes(raw_path.read_bytes())
        if existing != digest:
            raise HTTPException(status_code=409, detail={"code": "WH-RAW-IMMUTABLE", "message": "raw file already exists with different hash"})
    else:
        raw_path.write_bytes(data)
    return {
        "raw_text_ref": _relative_to_root(raw_path, paths["warehouse_root"]),
        "raw_hash": digest,
        "raw_size_bytes": len(data),
    }


def _source_locator_hash(req: WarehouseCreateItemReq) -> str:
    locator = req.raw_pointer or req.provenance.source_locator or req.provenance.source_title
    return _hash_text(locator or req.title)


def _review_id(item_id: str, reviewer_id: str) -> str:
    return _id("REV", item_id + reviewer_id + _now())


def _approval_id(item_id: str, approver_id: str) -> str:
    return _id("AEV", item_id + approver_id + _now())


def _dry_run_id(item_id: str) -> str:
    return _id("DRY", item_id + _now())


def _trace_id(item_id: str) -> str:
    return _id("PTR", item_id + _now())


def _backup_id() -> str:
    return _id("BAK", _now())


def _confidence_label(score: float) -> str:
    return "high" if score >= 0.85 else "low_confidence"


def _issue(code: str, severity: str, message: str) -> Dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def _decision_from_issues(issues: List[Dict[str, str]]) -> str:
    if not issues:
        return "PASS"
    if any(i.get("severity") == "BLOCKER" for i in issues):
        return "HOLD"
    return "NOT_VERIFIED"


def _validator_result(name: str, target: str, issues: List[Dict[str, str]]) -> Dict[str, Any]:
    decision = _decision_from_issues(issues)
    return {
        "ok": decision == "PASS",
        "validator": name,
        "target": target,
        "checked_at": _now(),
        "issues": issues,
        "decision": decision,
    }


def _validate_manifest() -> Dict[str, Any]:
    paths = _ensure_layout()
    issues = []
    manifest = _read_json(paths["warehouse_root"] / "warehouse_manifest.json", {})
    required_roots = manifest.get("roots") if isinstance(manifest, dict) else None
    if not isinstance(required_roots, dict):
        issues.append(_issue("MANIFEST_ROOTS_MISSING", "BLOCKER", "manifest roots are required"))
    else:
        for key, value in required_roots.items():
            if not Path(str(value)).exists():
                issues.append(_issue("MANIFEST_ROOT_MISSING", "BLOCKER", f"root does not exist: {key}"))
    if manifest.get("module_id") != "QLIB-WAREHOUSE":
        issues.append(_issue("MANIFEST_MODULE_INVALID", "BLOCKER", "module_id must be QLIB-WAREHOUSE"))
    return _validator_result("validate_manifest", "warehouse_manifest.json", issues)


def _validate_raw_hash(item: Dict[str, Any]) -> Dict[str, Any]:
    paths = _paths()
    issues = []
    raw_ref = str(item.get("raw_text_ref") or "")
    raw_hash = str(item.get("raw_hash") or "")
    if not raw_ref:
        issues.append(_issue("RAW_REF_MISSING", "BLOCKER", "raw_text_ref is required"))
    if not raw_hash.startswith("sha256:"):
        issues.append(_issue("RAW_HASH_MISSING", "BLOCKER", "raw_hash is required"))
    raw_path = paths["warehouse_root"] / raw_ref
    if raw_ref and not raw_path.exists():
        issues.append(_issue("RAW_FILE_MISSING", "BLOCKER", "raw file does not exist"))
    elif raw_ref and raw_hash and _hash_bytes(raw_path.read_bytes()) != raw_hash:
        issues.append(_issue("RAW_HASH_MISMATCH", "BLOCKER", "raw_hash does not match raw file"))
    return _validator_result("validate_raw_hash", str(item.get("warehouse_item_id") or ""), issues)


def _validate_provenance(item: Dict[str, Any]) -> Dict[str, Any]:
    issues = []
    prov = item.get("provenance")
    if not isinstance(prov, dict):
        issues.append(_issue("PROVENANCE_MISSING", "BLOCKER", "provenance is required"))
    else:
        for field in ["source_type", "source_title", "captured_by", "captured_at", "source_locator_hash"]:
            if not str(prov.get(field) or "").strip():
                issues.append(_issue("PROVENANCE_FIELD_MISSING", "BLOCKER", f"provenance.{field} is required"))
    return _validator_result("validate_provenance", str(item.get("warehouse_item_id") or ""), issues)


def _validate_rights(item: Dict[str, Any]) -> Dict[str, Any]:
    issues = []
    rights = str(item.get("rights_status") or "")
    if rights not in RIGHTS_STATUSES:
        issues.append(_issue("RIGHTS_INVALID", "BLOCKER", "rights_status enum is invalid"))
    if rights == "unknown":
        issues.append(_issue("RIGHTS_UNKNOWN", "BLOCKER", "rights_status unknown blocks approval and promotion"))
    return _validator_result("validate_rights", str(item.get("warehouse_item_id") or ""), issues)


def _validate_sensitivity(item: Dict[str, Any]) -> Dict[str, Any]:
    issues = []
    sensitivity = str(item.get("sensitivity") or "")
    if sensitivity not in SENSITIVITIES:
        issues.append(_issue("SENSITIVITY_INVALID", "BLOCKER", "sensitivity enum is invalid"))
    if sensitivity == "secret":
        issues.append(_issue("SECRET_BLOCKED", "BLOCKER", "secret items cannot be approved or promoted"))
    return _validator_result("validate_sensitivity", str(item.get("warehouse_item_id") or ""), issues)


def _validate_review(item: Dict[str, Any]) -> Dict[str, Any]:
    issues = []
    reviews = item.get("reviews") or []
    if item.get("status") in {"approved_for_library", "promoted"} and not reviews:
        issues.append(_issue("REVIEW_MISSING", "BLOCKER", "approved/promoted items require review"))
    if reviews:
        latest = reviews[-1]
        for field in ["reviewer_id", "review_note", "review_decision"]:
            if not str(latest.get(field) or "").strip():
                issues.append(_issue("REVIEW_FIELD_MISSING", "BLOCKER", f"review.{field} is required"))
    return _validator_result("validate_review", str(item.get("warehouse_item_id") or ""), issues)


def _validate_quality(item: Dict[str, Any]) -> Dict[str, Any]:
    issues = []
    score = int(item.get("quality_score") or 0)
    if item.get("status") in {"approved_for_library", "promoted"} and score < 80:
        issues.append(_issue("QUALITY_BELOW_THRESHOLD", "BLOCKER", "quality_score must be >= 80"))
    if item.get("confidence") == "low_confidence":
        issues.append(_issue("LOW_CONFIDENCE", "BLOCKER", "low confidence requires human review before promotion"))
    return _validator_result("validate_quality", str(item.get("warehouse_item_id") or ""), issues)


def _validate_approval(item: Dict[str, Any]) -> Dict[str, Any]:
    issues = []
    if item.get("status") in {"approved_for_library", "promoted"}:
        approval = item.get("approval") or {}
        for field in ["approval_event_id", "approver_id", "approval_note", "approval_date"]:
            if not str(approval.get(field) or "").strip():
                issues.append(_issue("APPROVAL_FIELD_MISSING", "BLOCKER", f"approval.{field} is required"))
    return _validator_result("validate_approval", str(item.get("warehouse_item_id") or ""), issues)


def _validate_item_schema(item: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "warehouse_item_id",
        "source_id",
        "item_type",
        "status",
        "title",
        "summary",
        "raw_text_ref",
        "raw_hash",
        "raw_mime_type",
        "raw_size_bytes",
        "provenance",
        "rights_status",
        "sensitivity",
        "visibility",
        "quality_score",
        "confidence",
        "created_at",
        "updated_at",
    ]
    issues = []
    for field in required:
        if field not in item:
            issues.append(_issue("ITEM_FIELD_MISSING", "BLOCKER", f"{field} is required"))
    if item.get("item_type") not in ITEM_TYPES:
        issues.append(_issue("ITEM_TYPE_INVALID", "BLOCKER", "item_type enum is invalid"))
    if item.get("status") not in STATUSES:
        issues.append(_issue("STATUS_INVALID", "BLOCKER", "status enum is invalid"))
    if item.get("visibility") not in VISIBILITIES:
        issues.append(_issue("VISIBILITY_INVALID", "BLOCKER", "visibility enum is invalid"))
    return _validator_result("validate_item_schema", str(item.get("warehouse_item_id") or ""), issues)


def _scan_text_for_secrets(text: str) -> List[Dict[str, str]]:
    issues = []
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            issues.append(_issue("SECRET_PATTERN_DETECTED", "BLOCKER", "secret-like pattern detected"))
            break
    return issues


def _validate_security_scan(item: Dict[str, Any]) -> Dict[str, Any]:
    issues = []
    text = json.dumps(
        {
            "title": item.get("title"),
            "summary": item.get("summary"),
            "provenance": item.get("provenance"),
            "reviews": item.get("reviews"),
        },
        ensure_ascii=False,
    )
    issues.extend(_scan_text_for_secrets(text))
    return _validator_result("validate_security_scan", str(item.get("warehouse_item_id") or ""), issues)


def _all_item_validators(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        _validate_item_schema(item),
        _validate_raw_hash(item),
        _validate_provenance(item),
        _validate_rights(item),
        _validate_sensitivity(item),
        _validate_review(item),
        _validate_quality(item),
        _validate_approval(item),
        _validate_security_scan(item),
    ]


def _blocking_issues(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    for result in results:
        for issue in result.get("issues") or []:
            if issue.get("severity") == "BLOCKER":
                issues.append(issue)
    return issues


def _hard_gate_results(item: Dict[str, Any]) -> Dict[str, Any]:
    validators = _all_item_validators(item)
    blockers = _blocking_issues(validators)
    return {
        "ok": not blockers,
        "decision": "PASS" if not blockers else "HOLD",
        "validators": validators,
        "issues": blockers,
    }


def _item_hash(item: Dict[str, Any]) -> str:
    stable = dict(item)
    stable.pop("updated_at", None)
    return _hash_text(json.dumps(stable, ensure_ascii=False, sort_keys=True))


def _latest_backup_manifest() -> Optional[Dict[str, Any]]:
    paths = _paths()
    manifests = sorted(paths["backup_root"].glob("*/backup_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not manifests:
        return None
    return _read_json(manifests[0], None)


def _release_board_path() -> Path:
    return _paths()["release_root"] / "release_board.json"


def _run_w_gates(item: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    manifest = _validate_manifest()
    latest_backup = _latest_backup_manifest()
    release_board = _read_json(_release_board_path(), None)
    gates: Dict[str, Dict[str, Any]] = {
        "W-G1": {"name": "Manifest Gate", "decision": "PASS" if manifest["ok"] else "HOLD", "issues": manifest["issues"]},
        "W-G7": {
            "name": "Backup Gate",
            "decision": "PASS" if latest_backup and latest_backup.get("restore_dry_run_pass") is True else "NOT_VERIFIED",
            "issues": [] if latest_backup and latest_backup.get("restore_dry_run_pass") is True else [_issue("BACKUP_RESTORE_NOT_VERIFIED", "MEDIUM", "restore dry-run is required")],
        },
        "W-G9": {
            "name": "Release Gate",
            "decision": "PASS" if isinstance(release_board, dict) and release_board.get("decision") in {"PASS", "HOLD"} else "NOT_VERIFIED",
            "issues": [] if isinstance(release_board, dict) else [_issue("RELEASE_BOARD_MISSING", "MEDIUM", "release board has not been updated")],
        },
    }
    if item:
        raw = _validate_raw_hash(item)
        prov = _validate_provenance(item)
        rights = _validate_rights(item)
        sensitivity = _validate_sensitivity(item)
        review = _validate_review(item)
        quality = _validate_quality(item)
        approval = _validate_approval(item)
        security = _validate_security_scan(item)
        promotion = item.get("promotion") or {}
        dry_run_pass = bool(promotion.get("promotion_dry_run_pass"))
        trace_pass = bool(promotion.get("promotion_trace_id"))
        library_engine_required = _library_engine_enabled() or bool(promotion.get("library_engine_enabled"))
        library_engine_pass = (not library_engine_required) or promotion.get("library_engine_decision") == "PASS"
        gates.update(
            {
                "W-G2": {"name": "Raw Gate", "decision": "PASS" if raw["ok"] else "HOLD", "issues": raw["issues"]},
                "W-G3": {
                    "name": "Provenance Gate",
                    "decision": "PASS" if prov["ok"] and rights["ok"] and sensitivity["ok"] else "HOLD",
                    "issues": (prov["issues"] + rights["issues"] + sensitivity["issues"]),
                },
                "W-G4": {
                    "name": "Review Gate",
                    "decision": "PASS" if review["ok"] and quality["ok"] else "HOLD",
                    "issues": review["issues"] + quality["issues"],
                },
                "W-G5": {"name": "Approval Gate", "decision": "PASS" if approval["ok"] else "HOLD", "issues": approval["issues"]},
                "W-G6": {
                    "name": "Promotion Gate",
                    "decision": "PASS" if dry_run_pass and trace_pass and library_engine_pass else ("NOT_IN_SCOPE" if item.get("status") != "promoted" else "HOLD"),
                    "issues": []
                    if dry_run_pass and trace_pass and library_engine_pass
                    else [_issue("PROMOTION_TRACE_NOT_COMPLETE", "MEDIUM", "dry-run, trace, and required library engine verification are required")],
                },
                "W-G8": {"name": "Security Gate", "decision": "PASS" if security["ok"] else "HOLD", "issues": security["issues"]},
            }
        )
    return gates


def _public_item(item: Dict[str, Any], include_raw: bool = False) -> Dict[str, Any]:
    out = dict(item)
    if not include_raw:
        out.pop("raw_text", None)
    if out.get("sensitivity") in {"private", "secret"} or out.get("visibility") in {"reviewer_only", "no_export"}:
        out["summary"] = "[masked]"
    return out


@router.get("/manifest")
def warehouse_manifest() -> Dict[str, Any]:
    paths = _ensure_layout()
    return {"ok": True, "manifest": _read_json(paths["warehouse_root"] / "warehouse_manifest.json", {})}


@router.get("/status")
def warehouse_status() -> Dict[str, Any]:
    _ensure_layout()
    items = list(_latest_items().values())
    counts: Dict[str, int] = {status: 0 for status in sorted(STATUSES)}
    for item in items:
        status = str(item.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    gates = _run_w_gates()
    return {
        "ok": True,
        "module_id": "QLIB-WAREHOUSE",
        "total_items": len(items),
        "state_counts": counts,
        "gate_results": gates,
        "backup_latest": _latest_backup_manifest(),
        "release_board_path": str(_release_board_path()),
    }


@router.post("/items", status_code=201)
def create_item(payload: WarehouseCreateItemReq) -> Dict[str, Any]:
    paths = _ensure_layout()
    if not payload.title.strip():
        raise HTTPException(status_code=422, detail={"code": "WH-ITEM-001", "message": "title is required"})
    data = _raw_payload(payload)
    seed = payload.title + payload.provenance.source_title + _hash_bytes(data) + _now()
    source_id = _id("SRC", seed)
    item_id = _id("WHI", seed)
    raw_meta = _write_raw_file(paths, item_id, data, payload.raw_mime_type)
    prov = _dump_model(payload.provenance)
    prov["captured_at"] = prov.get("captured_at") or _now()
    prov["source_locator_hash"] = _source_locator_hash(payload)
    now = _now()
    item = {
        "schema_version": 1,
        "contract_version": "1.0.0",
        "revision": 1,
        "source_id": source_id,
        "warehouse_item_id": item_id,
        "item_type": payload.item_type,
        "status": "captured",
        "title": payload.title.strip(),
        "summary": payload.summary.strip(),
        "raw_text_ref": raw_meta["raw_text_ref"],
        "raw_hash": raw_meta["raw_hash"],
        "raw_mime_type": payload.raw_mime_type,
        "raw_size_bytes": raw_meta["raw_size_bytes"],
        "provenance": prov,
        "rights_status": payload.rights_status,
        "sensitivity": payload.sensitivity,
        "visibility": payload.visibility,
        "tags": payload.tags,
        "quality_score": 0,
        "confidence": "unreviewed",
        "reviews": [],
        "approval": {},
        "promotion": {},
        "status_events": [{"from": None, "to": "captured", "actor_id": prov.get("captured_by"), "reason": "capture", "at": now}],
        "created_at": now,
        "updated_at": now,
    }
    _append_jsonl(_items_path(), item)
    validation = _validate_item_schema(item)
    return {"ok": True, "item": _public_item(item), "validation": validation}


@router.get("/items")
def list_items(
    status: Optional[str] = Query(default=None),
    item_type: Optional[str] = Query(default=None),
    include_private: bool = Query(default=False),
) -> Dict[str, Any]:
    _ensure_layout()
    items = []
    for item in _latest_items().values():
        if status and item.get("status") != status:
            continue
        if item_type and item.get("item_type") != item_type:
            continue
        if not include_private and item.get("visibility") in {"reviewer_only", "no_export"}:
            continue
        items.append(_public_item(item))
    items.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
    return {"ok": True, "count": len(items), "items": items}


@router.get("/items/{item_id}")
def read_item(item_id: str, include_raw: bool = Query(default=False)) -> Dict[str, Any]:
    item = _get_item(item_id)
    return {"ok": True, "item": _public_item(item, include_raw=include_raw), "validators": _all_item_validators(item)}


@router.patch("/items/{item_id}/status")
def update_status(item_id: str, payload: WarehouseStatusReq) -> Dict[str, Any]:
    item = _get_item(item_id)
    current = str(item.get("status") or "")
    target = payload.status
    if target in {"approved_for_library", "promoted"}:
        raise HTTPException(status_code=409, detail={"code": "WH-STATE-ENDPOINT", "message": "use approve or promote endpoint"})
    if target != current and target not in STATUS_TRANSITIONS.get(current, set()):
        raise HTTPException(status_code=409, detail={"code": "WH-STATE-INVALID", "message": f"transition not allowed: {current} -> {target}"})
    item["status"] = target
    item.setdefault("status_events", []).append({"from": current, "to": target, "actor_id": payload.actor_id, "reason": payload.reason, "at": _now()})
    item = _save_item(item)
    return {"ok": True, "item": _public_item(item), "transition": {"from": current, "to": target}}


@router.post("/items/{item_id}/reviews")
def add_review(item_id: str, payload: WarehouseReviewReq) -> Dict[str, Any]:
    item = _get_item(item_id)
    review = _dump_model(payload)
    review.update(
        {
            "review_id": _review_id(item_id, payload.reviewer_id),
            "warehouse_item_id": item_id,
            "review_date": _now(),
            "confidence": _confidence_label(payload.confidence_score),
        }
    )
    item.setdefault("reviews", []).append(review)
    item["quality_score"] = payload.quality_score
    item["confidence"] = review["confidence"]
    if payload.review_decision in {"needs_source", "hold", "rejected"}:
        current = str(item.get("status") or "")
        item["status"] = payload.review_decision
        item.setdefault("status_events", []).append({"from": current, "to": payload.review_decision, "actor_id": payload.reviewer_id, "reason": "review decision", "at": _now()})
    item = _save_item(item)
    _write_json(_paths()["review_root"] / f"{review['review_id']}.json", review)
    return {"ok": True, "item": _public_item(item), "review": review, "validators": _all_item_validators(item)}


@router.post("/items/{item_id}/approve")
def approve_for_library(item_id: str, payload: WarehouseApproveReq) -> Dict[str, Any]:
    item = _get_item(item_id)
    current = str(item.get("status") or "")
    if current != "pending_review":
        raise HTTPException(status_code=409, detail={"code": "WH-APPROVAL-STATE", "message": "item must be pending_review before approval"})
    approval = {
        "approval_event_id": _approval_id(item_id, payload.approver_id),
        "warehouse_item_id": item_id,
        "approver_id": payload.approver_id,
        "approval_note": payload.approval_note,
        "approval_date": _now(),
        "decision": "approved_for_library",
    }
    gate = _hard_gate_results({**item, "status": "approved_for_library", "approval": approval})
    if not gate["ok"]:
        if any(issue["code"] == "RIGHTS_UNKNOWN" for issue in gate["issues"]):
            item["status"] = "needs_source"
            _save_item(item)
        raise HTTPException(status_code=409, detail={"code": "WH-APPROVAL-GATE", "message": "approval hard gate failed", "issues": gate["issues"]})
    item["approval"] = approval
    item["status"] = "approved_for_library"
    item.setdefault("status_events", []).append({"from": current, "to": "approved_for_library", "actor_id": payload.approver_id, "reason": "approval", "at": approval["approval_date"]})
    item = _save_item(item)
    return {"ok": True, "item": _public_item(item), "approval": approval, "gate": _hard_gate_results(item)}


def _dry_run_for_item(item: Dict[str, Any], req: PromotionDryRunReq) -> Dict[str, Any]:
    issues = []
    if item.get("status") != "approved_for_library":
        issues.append(_issue("PROMOTION_STATUS_INVALID", "BLOCKER", "status must be approved_for_library"))
    if req.promotion_target not in PROMOTION_TARGETS:
        issues.append(_issue("PROMOTION_TARGET_INVALID", "BLOCKER", "promotion target is invalid"))
    gate = _hard_gate_results(item)
    issues.extend(gate["issues"])
    paths = _paths()
    library_id = f"LIB-{item['warehouse_item_id'].lower()}"
    graph_node_id = f"GND-{item['warehouse_item_id'].lower()}"
    evidence_id = f"EVD-{item['warehouse_item_id'].lower()}"
    dry_id = _dry_run_id(item["warehouse_item_id"])
    library_engine = _library_engine_dry_run(item, req)
    issues.extend(library_engine.get("issues") or [])
    planned_artifacts = {
        "library_card_path": str(paths["library_card_root"] / f"{library_id}.json"),
        "graph_node_path": str(paths["library_graph_root"] / f"{graph_node_id}.json"),
        "evidence_card_path": str(paths["library_evidence_root"] / f"{evidence_id}.json"),
        "proofpack_path": str(paths["proofpack_root"] / f"{dry_id}.json"),
    }
    if library_engine.get("expected_artifacts"):
        planned_artifacts["qualilibrary_ripple"] = library_engine["expected_artifacts"]
    result = {
        "promotion_dry_run_id": dry_id,
        "warehouse_item_id": item["warehouse_item_id"],
        "promotion_target": req.promotion_target,
        "created_by": req.created_by,
        "created_at": _now(),
        "validation_result": {
            "manifest_pass": _validate_manifest()["ok"],
            "raw_pass": _validate_raw_hash(item)["ok"],
            "provenance_pass": _validate_provenance(item)["ok"],
            "rights_pass": _validate_rights(item)["ok"],
            "sensitivity_pass": _validate_sensitivity(item)["ok"],
            "review_pass": _validate_review(item)["ok"] and _validate_quality(item)["ok"],
            "dry_run_pass": not issues,
            "library_engine_pass": library_engine.get("decision") == "PASS",
            "library_engine_status": library_engine.get("decision"),
        },
        "policy_result": {
            "no_secret": item.get("sensitivity") != "secret",
            "no_paid_standard_long_quote": True,
            "no_private_export": item.get("visibility") not in {"no_export", "reviewer_only"},
            "no_internal_path_public_output": True,
        },
        "planned_artifacts": planned_artifacts,
        "library_engine": library_engine,
        "issues": issues,
        "decision": "PASS" if not issues else "HOLD",
    }
    return result


@router.post("/items/{item_id}/promotion-dry-run")
def promotion_dry_run(item_id: str, payload: PromotionDryRunReq) -> Dict[str, Any]:
    _ensure_layout()
    item = _get_item(item_id)
    result = _dry_run_for_item(item, payload)
    paths = _paths()
    _write_json(paths["dry_run_root"] / f"{result['promotion_dry_run_id']}.json", result)
    if result["decision"] == "PASS":
        item.setdefault("promotion", {})
        item["promotion"]["promotion_target"] = payload.promotion_target
        item["promotion"]["promotion_dry_run_id"] = result["promotion_dry_run_id"]
        item["promotion"]["promotion_dry_run_pass"] = True
        item = _save_item(item)
    return {"ok": result["decision"] == "PASS", "dry_run": result, "item": _public_item(item)}


@router.post("/items/{item_id}/promote")
def promote(item_id: str, payload: PromoteReq) -> Dict[str, Any]:
    paths = _ensure_layout()
    item = _get_item(item_id)
    dry_id = payload.promotion_dry_run_id or (item.get("promotion") or {}).get("promotion_dry_run_id")
    if not dry_id:
        raise HTTPException(status_code=409, detail={"code": "WH-PROMOTION-DRYRUN-MISSING", "message": "promotion dry-run is required"})
    dry_run = _read_json(paths["dry_run_root"] / f"{dry_id}.json", None)
    if not isinstance(dry_run, dict) or dry_run.get("decision") != "PASS":
        raise HTTPException(status_code=409, detail={"code": "WH-PROMOTION-DRYRUN-NOT-PASS", "message": "promotion dry-run must PASS"})
    if item.get("status") != "approved_for_library":
        raise HTTPException(status_code=409, detail={"code": "WH-PROMOTION-STATE", "message": "item must be approved_for_library"})

    library_engine = _library_engine_promote(item, payload, dry_run)
    trace_id = _trace_id(item_id)
    library_id = f"LIB-{item_id.lower()}"
    graph_node_id = f"GND-{item_id.lower()}"
    evidence_id = f"EVD-{item_id.lower()}"
    card_path = paths["library_card_root"] / f"{library_id}.json"
    graph_path = paths["library_graph_root"] / f"{graph_node_id}.json"
    evidence_path = paths["library_evidence_root"] / f"{evidence_id}.json"
    proofpack_path = paths["proofpack_root"] / f"{trace_id}.json"

    library_card = {
        "library_id": library_id,
        "source": "QLIB-WAREHOUSE",
        "title": item["title"],
        "summary": item["summary"],
        "promotion_target": payload.promotion_target,
        "source_trace": {
            "promotion_trace_id": trace_id,
            "warehouse_item_id": item_id,
            "source_id": item.get("source_id"),
            "raw_hash": item.get("raw_hash"),
        },
        "qualilibrary_ripple": {
            "enabled": library_engine.get("enabled"),
            "node_id": library_engine.get("node_id"),
            "memory_item_id": library_engine.get("memory_item_id"),
        },
        "raw_export_allowed": False,
        "created_at": _now(),
    }
    graph_node = {
        "graph_node_id": graph_node_id,
        "library_id": library_id,
        "node_type": payload.promotion_target,
        "title": item["title"],
        "created_at": _now(),
    }
    evidence_card = {
        "evidence_id": evidence_id,
        "library_id": library_id,
        "warehouse_item_id": item_id,
        "raw_hash": item.get("raw_hash"),
        "raw_text_ref": item.get("raw_text_ref"),
        "pointer_only": True,
        "created_at": _now(),
    }
    validation_result = dict(dry_run.get("validation_result") or {})
    validation_result["library_engine_pass"] = library_engine.get("decision") == "PASS"
    validation_result["library_engine_status"] = library_engine.get("decision")
    policy_result = dry_run.get("policy_result") or {}
    output_artifacts = {
        "library_card_path": str(card_path),
        "graph_node_path": str(graph_path),
        "evidence_card_path": str(evidence_path),
        "proofpack_path": str(proofpack_path),
    }
    if library_engine.get("expected_artifacts"):
        output_artifacts["qualilibrary_ripple"] = library_engine["expected_artifacts"]
    trace = {
        "promotion_trace_id": trace_id,
        "warehouse_item_id": item_id,
        "warehouse_item_hash": _item_hash(item),
        "raw_hash": item.get("raw_hash"),
        "source_item_status": "approved_for_library",
        "promotion_target": payload.promotion_target,
        "promoted_library_id": library_id,
        "promoted_graph_node_id": graph_node_id,
        "promoted_evidence_ids": [evidence_id],
        "validation_result": validation_result,
        "policy_result": policy_result,
        "output_artifacts": output_artifacts,
        "library_engine": library_engine,
        "created_by": payload.created_by,
        "created_at": _now(),
    }
    _write_json(card_path, library_card)
    _write_json(graph_path, graph_node)
    _write_json(evidence_path, evidence_card)
    _write_json(proofpack_path, {"trace": trace, "dry_run": dry_run, "gate_results": _run_w_gates(item)})
    _append_jsonl(_trace_path(), trace)

    current = item.get("status")
    item["status"] = "promoted"
    item.setdefault("promotion", {}).update(
        {
            "promotion_target": payload.promotion_target,
            "promotion_dry_run_id": dry_id,
            "promotion_dry_run_pass": True,
            "promoted_library_id": library_id,
            "promoted_graph_node_id": graph_node_id,
            "promoted_evidence_ids": [evidence_id],
            "promotion_trace_id": trace_id,
            "library_engine_enabled": library_engine.get("enabled"),
            "library_engine_decision": library_engine.get("decision"),
            "library_node_id": library_engine.get("node_id"),
            "library_memory_item_id": library_engine.get("memory_item_id"),
            "library_artifacts": library_engine.get("expected_artifacts") or {},
        }
    )
    item.setdefault("status_events", []).append({"from": current, "to": "promoted", "actor_id": payload.created_by, "reason": "promotion trace sealed", "at": _now()})
    item = _save_item(item)
    return {"ok": True, "trace": trace, "item": _public_item(item)}


@router.get("/traces/{trace_id}")
def read_trace(trace_id: str) -> Dict[str, Any]:
    for row in _read_jsonl(_trace_path()):
        if row.get("promotion_trace_id") == trace_id:
            return {"ok": True, "trace": row}
    raise HTTPException(status_code=404, detail={"code": "WH-TRACE-404", "message": "promotion trace not found"})


@router.post("/validate")
def validate_warehouse(item_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    _ensure_layout()
    if item_id:
        item = _get_item(item_id)
        validators = [_validate_manifest()] + _all_item_validators(item)
        gate_results = _run_w_gates(item)
        ok = all(v["ok"] for v in validators) and all(g["decision"] in {"PASS", "NOT_IN_SCOPE"} for g in gate_results.values())
        return {"ok": ok, "target": item_id, "validators": validators, "gate_results": gate_results}
    validators = [_validate_manifest()]
    items = list(_latest_items().values())
    item_results = [{"warehouse_item_id": item["warehouse_item_id"], "validators": _all_item_validators(item)} for item in items]
    ok = validators[0]["ok"] and all(not _blocking_issues(r["validators"]) for r in item_results)
    return {"ok": ok, "target": "warehouse", "validators": validators, "items": item_results, "gate_results": _run_w_gates()}


def _iter_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


def _build_backup_manifest(paths: Dict[str, Path], backup_id: str, backup_dir: Path, created_by: str) -> Dict[str, Any]:
    source_roots = [
        paths["warehouse_root"],
        paths["proofpack_root"],
        paths["release_root"],
    ]
    files = []
    total = 0
    for root in source_roots:
        for file in _iter_files(root):
            total += file.stat().st_size
            files.append({"path": str(file), "sha256": _hash_bytes(file.read_bytes()), "bytes": file.stat().st_size})
    digest = _hash_text(json.dumps(files, sort_keys=True))
    return {
        "backup_id": backup_id,
        "backup_date": _now(),
        "backup_scope": "full_warehouse",
        "backup_path": str(backup_dir),
        "included_roots": [str(r) for r in source_roots],
        "file_count": len(files),
        "total_bytes": total,
        "sha256_manifest": digest,
        "created_by": created_by,
        "restore_dry_run_required": True,
        "restore_dry_run_pass": False,
    }


@router.post("/backup/run")
def backup_run(payload: BackupRunReq) -> Dict[str, Any]:
    paths = _ensure_layout()
    backup_id = _backup_id()
    backup_dir = paths["backup_root"] / backup_id
    manifest = _build_backup_manifest(paths, backup_id, backup_dir, payload.created_by)
    if payload.dry_run_only:
        manifest["dry_run_only"] = True
        return {"ok": True, "backup": manifest}
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ["warehouse_root", "proofpack_root", "release_root"]:
        src = paths[name]
        dst = backup_dir / name
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
    _write_json(backup_dir / "backup_manifest.json", manifest)
    return {"ok": True, "backup": manifest}


@router.post("/backup/restore-dry-run/{backup_id}")
def restore_dry_run(backup_id: str) -> Dict[str, Any]:
    paths = _ensure_layout()
    manifest_path = paths["backup_root"] / backup_id / "backup_manifest.json"
    manifest = _read_json(manifest_path, None)
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=404, detail={"code": "WH-BACKUP-404", "message": "backup manifest not found"})
    issues = []
    for root_name in ["warehouse_root", "proofpack_root", "release_root"]:
        if not (paths["backup_root"] / backup_id / root_name).exists():
            issues.append(_issue("BACKUP_ROOT_MISSING", "BLOCKER", f"backup root missing: {root_name}"))
    manifest["restore_dry_run_at"] = _now()
    manifest["restore_dry_run_pass"] = not issues
    manifest["restore_dry_run_issues"] = issues
    _write_json(manifest_path, manifest)
    return {"ok": not issues, "backup": manifest}


@router.get("/release-board")
def get_release_board() -> Dict[str, Any]:
    board = _read_json(_release_board_path(), None)
    if not isinstance(board, dict):
        return {"ok": True, "release_board": None, "decision": "NOT_VERIFIED"}
    return {"ok": True, "release_board": board, "decision": board.get("decision")}


@router.post("/release-board/update")
def update_release_board(payload: ReleaseBoardReq, item_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    _ensure_layout()
    item = _get_item(item_id) if item_id else None
    gates = _run_w_gates(item)
    gates["W-G9"] = {"name": "Release Gate", "decision": "PASS", "issues": []}
    hard_fail = any(g["decision"] == "HOLD" for g in gates.values())
    not_verified = any(g["decision"] == "NOT_VERIFIED" for g in gates.values())
    decision = "HOLD" if hard_fail or not_verified else "PASS"
    latest_backup = _latest_backup_manifest()
    board = {
        "release_id": _id("REL", payload.scope + _now()),
        "date": _now(),
        "scope": payload.scope,
        "changed_files": payload.changed_files,
        "validators": ["validate_manifest", "validate_item_schema", "validate_raw_hash", "validate_provenance", "validate_rights", "validate_security_scan"],
        "gate_results": gates,
        "test_results": payload.test_results,
        "backup_id": latest_backup.get("backup_id") if isinstance(latest_backup, dict) else None,
        "rollback_plan": payload.rollback_plan,
        "decision": decision,
        "approver": payload.approver,
        "handover_path": payload.handover_path,
    }
    _write_json(_release_board_path(), board)
    return {"ok": True, "release_board": board}
