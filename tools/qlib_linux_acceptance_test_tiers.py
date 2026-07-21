"""Run the exact D42C platform tiers without hiding legacy failures.

Native Win32-only cases remain executable diagnostics.  Dual-platform tests
whose Windows branch asserts the retired native launcher stay required on
Linux and are separated only from the Windows portable tier.  This runner
persists counts, enums, and digests; unittest tracebacks and raw runtime values
remain in memory and are never written to the evidence bundle.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "R9ZNW-488D42C_PLATFORM_TIER_EVIDENCE_V1"
MANIFEST_SCHEMA_VERSION = "R9ZNW-488D42C_TEST_TIERS_V1"
LINUX_TIER = "LINUX_SHARED_REQUIRED"
WINDOWS_TIER = "WINDOWS_PORTABLE_SHARED"
WINDOWS_NATIVE_VERDICT = "WINDOWS_NATIVE_OBSERVER_NOT_APPROVED"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXACT_MODULE_ORDER = (
    "tools.test_qlib_linux_process_supervisor",
    "tools.test_qlib_traffic_observer",
    "tools.test_qlib_observer_evidence_contract",
    "tools.test_qlib_production_sampler",
)
ROOT_KEYS = frozenset(
    {
        "schema_version",
        "module_order",
        "expected_discovered_test_count",
        "expected_discovered_test_ids_digest",
        "required_shared_test_ids",
        "legacy_native_win32_test_ids",
        "windows_native_public_legacy_expectation_test_ids",
        "windows_portable_linux_only_exclusions",
        "tiers",
        "windows_native_support_verdict",
    }
)
SKIP_OR_XFAIL_DECORATORS = frozenset(
    {"skip", "skipIf", "skipUnless", "expectedFailure"}
)


class TierGateError(RuntimeError):
    """Fail-closed platform-tier validation error with an allowlisted cause."""

    def __init__(self, cause: str) -> None:
        super().__init__(cause)
        self.cause = cause


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _id_digest(test_ids: Iterable[str]) -> str:
    return _digest(sorted(test_ids))


def _require_string_list(
    manifest: Mapping[str, object], key: str, expected_count: int
) -> tuple[str, ...]:
    value = manifest.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TierGateError("MANIFEST_SCHEMA_INVALID")
    result = tuple(value)
    if len(result) != expected_count or len(set(result)) != expected_count:
        raise TierGateError("MANIFEST_CARDINALITY_INVALID")
    if result != tuple(sorted(result)):
        raise TierGateError("MANIFEST_ORDER_INVALID")
    return result


def _load_manifest(
    path: Path,
) -> tuple[
    dict[str, object],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    int,
    str,
]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TierGateError("MANIFEST_READ_INVALID") from exc
    if not isinstance(manifest, dict) or frozenset(manifest) != ROOT_KEYS:
        raise TierGateError("MANIFEST_SCHEMA_INVALID")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise TierGateError("MANIFEST_SCHEMA_INVALID")
    if tuple(manifest.get("module_order", ())) != EXACT_MODULE_ORDER:
        raise TierGateError("MODULE_SET_INVALID")
    if manifest.get("windows_native_support_verdict") != WINDOWS_NATIVE_VERDICT:
        raise TierGateError("WINDOWS_SUPPORT_VERDICT_INVALID")

    expected_discovered_count = manifest.get("expected_discovered_test_count")
    expected_discovered_digest = manifest.get("expected_discovered_test_ids_digest")
    if (
        not isinstance(expected_discovered_count, int)
        or expected_discovered_count <= 0
        or not isinstance(expected_discovered_digest, str)
        or len(expected_discovered_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_discovered_digest)
    ):
        raise TierGateError("DISCOVERY_BINDING_INVALID")
    required_shared_ids = _require_string_list(
        manifest, "required_shared_test_ids", expected_count=14
    )

    legacy_ids = _require_string_list(
        manifest, "legacy_native_win32_test_ids", expected_count=8
    )
    windows_public_legacy_ids = _require_string_list(
        manifest,
        "windows_native_public_legacy_expectation_test_ids",
        expected_count=5,
    )
    linux_only_ids = _require_string_list(
        manifest, "windows_portable_linux_only_exclusions", expected_count=9
    )
    groups = (
        set(required_shared_ids),
        set(legacy_ids),
        set(windows_public_legacy_ids),
        set(linux_only_ids),
    )
    if any(
        groups[left].intersection(groups[right])
        for left in range(4)
        for right in range(left + 1, 4)
    ):
        raise TierGateError("MANIFEST_GROUP_OVERLAP")

    tiers = manifest.get("tiers")
    expected_tiers = {
        LINUX_TIER: {
            "os_name": "posix",
            "sys_platform": "linux",
            "excluded_groups": ["legacy_native_win32_test_ids"],
        },
        WINDOWS_TIER: {
            "os_name": "nt",
            "sys_platform": "win32",
            "excluded_groups": [
                "legacy_native_win32_test_ids",
                "windows_native_public_legacy_expectation_test_ids",
                "windows_portable_linux_only_exclusions",
            ],
        },
    }
    if tiers != expected_tiers:
        raise TierGateError("TIER_CONTRACT_INVALID")
    return (
        manifest,
        required_shared_ids,
        legacy_ids,
        windows_public_legacy_ids,
        linux_only_ids,
        expected_discovered_count,
        expected_discovered_digest,
    )


def _flatten(suite: unittest.TestSuite) -> Iterator[unittest.TestCase]:
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        elif isinstance(item, unittest.TestCase):
            yield item
        else:
            raise TierGateError("DISCOVERY_OBJECT_INVALID")


def _load_exact_modules(
) -> tuple[tuple[ModuleType, ...], tuple[unittest.TestCase, ...]]:
    repository_root = str(REPOSITORY_ROOT)
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    modules: list[ModuleType] = []
    tests: list[unittest.TestCase] = []
    loader = unittest.TestLoader()
    for module_name in EXACT_MODULE_ORDER:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise TierGateError("TEST_MODULE_IMPORT_FAILED") from exc
        modules.append(module)
        tests.extend(_flatten(loader.loadTestsFromModule(module)))
    if loader.errors:
        raise TierGateError("TEST_DISCOVERY_FAILED")
    test_ids = [test.id() for test in tests]
    if len(test_ids) != len(set(test_ids)):
        raise TierGateError("DUPLICATE_TEST_ID")
    return tuple(modules), tuple(tests)


def _decorator_name(node: ast.expr) -> str:
    while isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return "UNKNOWN"


def _prove_excluded_tests_are_executable(
    modules: Sequence[ModuleType], excluded_ids: Sequence[str]
) -> tuple[int, int]:
    modules_by_name = {module.__name__: module for module in modules}
    trees: dict[str, ast.Module] = {}
    static_locations: set[str] = set()
    runtime_locations: set[str] = set()
    for test_id in excluded_ids:
        try:
            module_name, class_name, method_name = test_id.rsplit(".", 2)
        except ValueError as exc:
            raise TierGateError("LEGACY_TEST_ID_INVALID") from exc
        module = modules_by_name.get(module_name)
        if module is None or not module.__file__:
            raise TierGateError("LEGACY_TEST_SOURCE_MISSING")
        if module_name not in trees:
            try:
                trees[module_name] = ast.parse(
                    Path(module.__file__).read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, SyntaxError) as exc:
                raise TierGateError("LEGACY_TEST_SOURCE_INVALID") from exc
        class_node = next(
            (
                node
                for node in trees[module_name].body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ),
            None,
        )
        if class_node is None:
            raise TierGateError("LEGACY_TEST_CLASS_MISSING")
        method_node = next(
            (
                node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == method_name
            ),
            None,
        )
        if method_node is None:
            raise TierGateError("LEGACY_TEST_METHOD_MISSING")
        for owner, decorators in (
            (f"{module_name}.{class_name}", class_node.decorator_list),
            (test_id, method_node.decorator_list),
        ):
            for decorator in decorators:
                if _decorator_name(decorator) in SKIP_OR_XFAIL_DECORATORS:
                    static_locations.add(owner)

        test_class = getattr(module, class_name, None)
        if not isinstance(test_class, type):
            raise TierGateError("LEGACY_TEST_CLASS_MISSING")
        method = getattr(test_class, method_name, None)
        if method is None:
            raise TierGateError("LEGACY_TEST_METHOD_MISSING")
        for owner, target in (
            (f"{module_name}.{class_name}", test_class),
            (test_id, method),
        ):
            if bool(getattr(target, "__unittest_skip__", False)) or bool(
                getattr(target, "__unittest_expecting_failure__", False)
            ):
                runtime_locations.add(owner)

    decorator_count = len(static_locations)
    runtime_flag_count = len(runtime_locations)
    if decorator_count or runtime_flag_count:
        raise TierGateError("LEGACY_TEST_DECORATED")
    return decorator_count, runtime_flag_count


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, ensure_ascii=True, indent=2, sort_keys=True
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sealed(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["report_digest"] = _digest(payload)
    return result


def _run(manifest_path: Path, output: Path) -> int:
    (
        manifest,
        required_shared_ids,
        legacy_ids,
        windows_public_legacy_ids,
        linux_only_ids,
        expected_discovered_count,
        expected_discovered_digest,
    ) = _load_manifest(manifest_path)
    modules, discovered_tests = _load_exact_modules()
    discovered_by_id = {test.id(): test for test in discovered_tests}
    discovered_digest = _id_digest(discovered_by_id)
    if (
        len(discovered_by_id) != expected_discovered_count
        or discovered_digest != expected_discovered_digest
    ):
        raise TierGateError("DISCOVERY_SET_MISMATCH")
    required_manifest_ids = (
        set(required_shared_ids)
        .union(legacy_ids)
        .union(windows_public_legacy_ids)
        .union(linux_only_ids)
    )
    if not required_manifest_ids.issubset(discovered_by_id):
        raise TierGateError("MANIFEST_TEST_ID_MISSING")

    preserved_windows_ids = set(legacy_ids).union(windows_public_legacy_ids)
    decorator_count, runtime_flag_count = _prove_excluded_tests_are_executable(
        modules, tuple(sorted(preserved_windows_ids))
    )
    if os.name == "posix" and sys.platform == "linux":
        platform_enum = "LINUX"
        tier = LINUX_TIER
        excluded_ids = set(legacy_ids)
    elif os.name == "nt" and sys.platform == "win32":
        platform_enum = "WINDOWS_NT"
        tier = WINDOWS_TIER
        excluded_ids = (
            set(legacy_ids)
            .union(windows_public_legacy_ids)
            .union(linux_only_ids)
        )
    else:
        raise TierGateError("UNSUPPORTED_PLATFORM")

    selected_tests = tuple(
        discovered_by_id[test_id]
        for test_id in sorted(discovered_by_id)
        if test_id not in excluded_ids
    )
    selected_test_ids = {test.id() for test in selected_tests}
    if not set(required_shared_ids).issubset(selected_test_ids):
        raise TierGateError("REQUIRED_SHARED_TEST_NOT_SELECTED")
    result = unittest.TestResult()
    unittest.TestSuite(selected_tests).run(result)

    failure_count = len(result.failures)
    error_count = len(result.errors)
    skipped_count = len(result.skipped)
    expected_failure_count = len(result.expectedFailures)
    unexpected_success_count = len(result.unexpectedSuccesses)
    tests_run_count = int(result.testsRun)
    selected_count = len(selected_tests)
    passed = (
        tests_run_count == selected_count
        and failure_count == 0
        and error_count == 0
        and skipped_count == 0
        and expected_failure_count == 0
        and unexpected_success_count == 0
    )

    regression = _sealed(
        {
            "schema_version": SCHEMA_VERSION,
            "tier": tier,
            "platform": platform_enum,
            "verdict": "PASS" if passed else "FAIL",
            "module_count": len(modules),
            "discovered_test_count": len(discovered_tests),
            "selected_test_count": selected_count,
            "excluded_test_count": len(excluded_ids),
            "tests_run_count": tests_run_count,
            "failure_count": failure_count,
            "error_count": error_count,
            "skipped_count": skipped_count,
            "expected_failure_count": expected_failure_count,
            "unexpected_success_count": unexpected_success_count,
            "manifest_digest": _digest(manifest),
            "discovered_test_ids_digest": discovered_digest,
            "required_shared_test_count": len(required_shared_ids),
            "required_shared_test_ids_digest": _id_digest(required_shared_ids),
            "selected_test_ids_digest": _id_digest(
                test.id() for test in selected_tests
            ),
            "excluded_test_ids_digest": _id_digest(excluded_ids),
        }
    )
    windows_native = _sealed(
        {
            "schema_version": SCHEMA_VERSION,
            "support_verdict": WINDOWS_NATIVE_VERDICT,
            "gate_membership": "SEPARATE_LEGACY_DIAGNOSTIC",
            "automatic_selection_state": "DISABLED",
            "required_linux_gate_state": "EXCLUDED_BY_EXACT_ID",
            "diagnostic_execution_state": "NOT_EXECUTED_IN_LINUX_ACCEPTANCE",
            "legacy_test_count": len(legacy_ids),
            "legacy_test_ids_digest": _id_digest(legacy_ids),
            "windows_native_public_legacy_expectation_test_count": len(
                windows_public_legacy_ids
            ),
            "windows_native_public_legacy_expectation_test_ids_digest": _id_digest(
                windows_public_legacy_ids
            ),
            "preserved_windows_native_test_count": len(legacy_ids)
            + len(windows_public_legacy_ids),
            "public_windows_guard_state": "FAIL_CLOSED_TEST_SELECTED_BY_EXACT_ID",
            "skip_or_xfail_decorator_count": decorator_count,
            "runtime_skip_or_xfail_flag_count": runtime_flag_count,
        }
    )
    _atomic_write(output / "linux-shared-regression.json", regression)
    _atomic_write(output / "windows-native-support.json", windows_native)
    print(f"platform_tier_status={'PASS' if passed else 'FAIL'}")
    print(f"platform_tier={tier}")
    print(f"tests_run_count={tests_run_count}")
    return 0 if passed else 1


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exact D42C portable test tier"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("qlib_linux_acceptance_test_tiers.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return _run(args.manifest.resolve(), args.output.resolve())
    except TierGateError as exc:
        print("platform_tier_status=FAIL")
        print(f"cause={exc.cause}")
        return 2
    except Exception:
        print("platform_tier_status=FAIL")
        print("cause=UNEXPECTED_FAIL_CLOSED")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
