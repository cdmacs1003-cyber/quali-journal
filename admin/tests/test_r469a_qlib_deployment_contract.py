import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "deploy" / "qlib-skillup-runtime"
CONTRACT_PATH = TARGET / "target-contract.json"
DOCKERFILE_PATH = TARGET / "Dockerfile"
IGNORE_PATH = TARGET / "Dockerfile.dockerignore"
LOCK_PATH = TARGET / "requirements.lock"
RUNBOOK_PATH = TARGET / "operations-runbook.md"
ARTIFACT_BUILD_SCRIPT_PATH = ROOT / "tools" / "build_qlib_runtime_artifact.ps1"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_target_contract_is_exact_private_no_data_change_target():
    contract = _contract()
    target = contract["target"]
    assert target["service"] == "qlib-skillup-runtime"
    assert target["region"] == "asia-northeast1"
    assert target["mode"] == "authenticated_limited_field_beta"
    assert target["public_access"] is False
    assert target["custom_domain"] == "DEFERRED_NOT_AUTHORIZED"
    assert target["protected_unrelated_service"] == "quali-admin-domap"
    assert target["protected_unrelated_service_action"] == "DO_NOT_DEPLOY_DO_NOT_MUTATE_DO_NOT_ROUTE_TRAFFIC"

    impact = contract["data_impact"]
    false_fields = (
        "production_db_read",
        "production_db_write",
        "schema_change",
        "migration",
        "production_library_write",
        "persistent_storage_change",
        "queue_or_scheduler_change",
        "external_provider_write",
        "raw_user_text_retention",
        "analytics_retention_change",
        "raw_standard_text_export",
        "unknown_rights_use",
        "cloud_sql_quali_pg_dependency",
    )
    assert all(impact[field] is False for field in false_fields)
    assert impact["bridge_evidence_trace_required"] is True


def test_target_contract_runtime_resource_secret_and_traffic_closure():
    contract = _contract()
    runtime = contract["runtime"]
    assert runtime["required_secret_bindings"] == []
    assert runtime["required_secret_binding_count"] == 0
    assert "private service" in runtime["authentication_enforcement"]
    assert len(runtime["required_deployer_permissions"]) == 7
    assert contract["resources"] == {
        "min_instances": 0,
        "max_instances": 2,
        "cpu": "1",
        "memory": "512Mi",
        "concurrency": 80,
        "timeout_seconds": 300,
        "cloud_sql_required": False,
    }
    assert contract["traffic"]["strategy_percent"] == [0, 5, 20, 50, 100]
    assert contract["traffic"]["initial_rollback_revision"] == "qlib-skillup-runtime-00002-d9g"
    assert contract["source"]["source_head"]["resolver"] == "git rev-parse HEAD"


def test_dockerfile_is_digest_pinned_nonroot_and_minimal():
    text = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert re.search(r"^FROM python:3\.11-slim@sha256:[0-9a-f]{64}$", text, re.MULTILINE)
    assert "--require-hashes" in text
    assert "--no-compile" in text
    assert "ARG SOURCE_DATE_EPOCH" in text
    assert 'find /opt/site-packages -exec touch -h -d "@${SOURCE_DATE_EPOCH}"' in text
    assert "tar --sort=name --format=posix" in text
    assert "--pax-option=delete=atime,delete=ctime" in text
    assert "COPY --from=dependencies --chown=0:0 /opt/site-packages.tar /opt/site-packages.tar" in text
    assert "PYTHONPATH=/tmp/qlib-site-packages" in text
    assert "useradd" not in text
    assert "USER 10001:10001" in text
    assert "HEALTHCHECK" in text
    assert "EXPOSE 8080" in text
    assert "COPY admin/ /app/admin/" not in text
    assert "COPY . " not in text
    for label in (
        "source_repository",
        "source_branch",
        "source_commit",
        "task_id",
        "target_service",
        "target_mode",
    ):
        assert f'{label}="' in text


def test_canonical_artifact_builder_enforces_clean_head_and_rewritten_timestamps():
    text = ARTIFACT_BUILD_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "git diff --quiet --" in text
    assert "git diff --cached --quiet --" in text
    assert "--no-cache" in text
    assert "--provenance=false" in text
    assert "--platform linux/amd64" in text
    assert "rewrite-timestamp=true" in text
    assert "SOURCE_DATE_EPOCH" in text
    assert '--builder $CanonicalBuilder' in text
    assert '$CanonicalBuilder = "desktop-linux"' in text
    assert '$ExpectedDockerDesktopVersion = "4.50.0.209931"' in text
    assert '$ExpectedDockerVersion = "28.5.1"' in text
    assert '$ExpectedBuildxVersion = "v0.29.1-desktop.1"' in text
    assert '$ExpectedBuildKitVersion = "v0.25.1"' in text
    assert "registry_write = $false" in text


def test_target_dockerignore_is_default_deny_with_sensitive_exclusions():
    lines = IGNORE_PATH.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "**"
    required = {
        ".env",
        ".env.*",
        "**/*secret*",
        "**/*credential*",
        "**/*token*",
        "**/*.pem",
        "**/*.key",
        "**/.git/**",
        "**/tests/**",
        "**/docs/**",
        "**/*.zip",
        "**/*.pdf",
    }
    assert required.issubset(set(lines))


def test_requirements_lock_is_exact_and_hash_required():
    text = LOCK_PATH.read_text(encoding="utf-8")
    records = [line for line in text.splitlines() if line and not line.startswith(("#", " "))]
    assert len(records) == 12
    assert all(re.match(r"^[a-z0-9-]+==[^ ]+ \\$", line) for line in records)
    assert text.count("--hash=sha256:") == 12
    assert "fastapi==0.115.4" in text
    assert "uvicorn==0.32.0" in text


def test_runtime_exposes_only_health_ui_and_existing_f13_router():
    spec = importlib.util.spec_from_file_location("r469a_qlib_runtime", TARGET / "runtime.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = TestClient(module.app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "service": "qlib-skillup-runtime"}
    page = client.get("/")
    assert page.status_code == 200
    assert "beta-minimal-form" in page.text
    paths = {route.path for route in module.app.routes}
    assert "/api/f13/bridge/skillup/bridge-answer" in paths
    assert "/api/logs" not in paths
    assert "/api/backup/status" not in paths


def test_runtime_answer_and_additional_review_flows_do_not_echo_query():
    spec = importlib.util.spec_from_file_location("r469a_qlib_runtime_flow", TARGET / "runtime.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = TestClient(module.app)

    answered = client.post(
        "/api/f13/bridge/skillup/bridge-answer",
        json={
            "requester_module": "Skillup",
            "ui_mode": "beta_minimal",
            "request_payload": {"question": "솔더링이란?"},
        },
    )
    assert answered.status_code == 200
    answer_body = answered.json()
    assert answer_body["result_status"] == "OK"
    assert answer_body["answer_status"] == "ANSWERED"
    assert answer_body["evidence"]
    assert answer_body["trace_id"]
    assert answer_body["raw_text_included"] is False
    assert answer_body["internal_path_included"] is False

    held_question = "승인된 근거가 없는 현장 질문"
    held = client.post(
        "/api/f13/bridge/skillup/bridge-answer",
        json={
            "requester_module": "Skillup",
            "ui_mode": "beta_minimal",
            "request_payload": {"question": held_question},
        },
    )
    assert held.status_code == 200
    hold_body = held.json()
    assert hold_body["result_status"] == "HOLD"
    assert hold_body["answer_status"] == "HOLD"
    assert held_question not in held.text
    assert hold_body["raw_text_included"] is False
    assert hold_body["internal_path_included"] is False


def test_asset_builder_is_byte_deterministic_in_isolated_tree(tmp_path, monkeypatch):
    import tools.build_assets_hash as builder

    ui = tmp_path / "admin"
    assets = ui / "assets"
    (assets / "nested").mkdir(parents=True)
    (ui / "index.html").write_bytes(
        b'<link href="assets/style.css"><img src="assets/nested/icon.svg">'
    )
    (assets / "style.css").write_bytes(b"body { color: #123; }\n")
    (assets / "nested" / "icon.svg").write_bytes(b"<svg></svg>\n")

    monkeypatch.setattr(builder, "UI_BASE", ui)
    monkeypatch.setattr(builder, "SRC_HTML", ui / "index.html")
    monkeypatch.setattr(builder, "SRC_DIR", assets)
    monkeypatch.setattr(builder, "DIST", ui / "dist")
    monkeypatch.setattr(builder, "DST_ASSETS", ui / "dist" / "assets")

    builder.clean_dist()
    builder.build()
    first = {
        path.relative_to(ui / "dist").as_posix(): path.read_bytes()
        for path in (ui / "dist").rglob("*")
        if path.is_file()
    }
    shutil.rmtree(ui / "dist")
    builder.clean_dist()
    builder.build()
    second = {
        path.relative_to(ui / "dist").as_posix(): path.read_bytes()
        for path in (ui / "dist").rglob("*")
        if path.is_file()
    }
    assert second == first


def test_source_and_dist_are_complete_manifest_normalized_twins():
    source = (ROOT / "admin" / "index.html").read_text(encoding="utf-8")
    dist = (ROOT / "admin" / "dist" / "index.html").read_text(encoding="utf-8")
    manifest = json.loads((ROOT / "admin" / "dist" / "manifest.json").read_text(encoding="utf-8"))
    for source_ref, dist_ref in manifest.items():
        dist = dist.replace(dist_ref, source_ref)
    assert dist == source

    expected = {}
    assets = ROOT / "admin" / "assets"
    for path in sorted(item for item in assets.rglob("*") if item.is_file()):
        rel = path.relative_to(assets).as_posix()
        stem, suffix = Path(rel).with_suffix("").as_posix(), path.suffix
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
        expected[f"assets/{rel}"] = f"assets/{stem}.{digest}{suffix}"
    assert manifest == expected
    expected_dist_files = {"index.html", "manifest.json"} | set(expected.values())
    actual_dist_files = {
        path.relative_to(ROOT / "admin" / "dist").as_posix()
        for path in (ROOT / "admin" / "dist").rglob("*")
        if path.is_file()
    }
    assert actual_dist_files == expected_dist_files


def test_runbook_is_target_specific_staged_and_nonexecuting():
    text = RUNBOOK_PATH.read_text(encoding="utf-8")
    assert "qlib-skillup-runtime" in text
    assert "asia-northeast1" in text
    assert "quali-admin-domap" in text and "must never target or mutate" in text
    assert "does not grant deployment authorization" in text
    for split in ("=5,", "=20,", "=50,", "=100"):
        assert split in text
    for duration in ("Observe 10 minutes", "Observe 15 minutes", "Final observation: 15 minutes"):
        assert duration in text
    for stop in (
        "health failure",
        "authentication-boundary regression",
        "Evidence/Trace failure",
        "unexpected Production write",
        "cost/capacity anomaly",
        "owner stop instruction",
    ):
        assert stop in text
    assert '"$STABLE_REVISION=100"' in text
