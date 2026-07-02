"""Raw-output-safe local listener verifier category helpers.

This module is import-safe and local-only. It does not start runtime, open
network connections, execute OS commands, inspect process tables, read
environment variables, access databases, inspect secrets, or print/log raw
listener rows. Callers provide already-captured in-memory rows and receive
sanitized category fields only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any


SOURCE_A_COMMAND_UNAVAILABLE = "source_a_command_unavailable"
SOURCE_A_COMMAND_AVAILABLE_NO_ROWS = "source_a_command_available_no_rows"
SOURCE_A_COMMAND_AVAILABLE_NO_MATCHING_ROWS = (
    "source_a_command_available_no_matching_rows"
)
SOURCE_A_COMMAND_AVAILABLE_PARSER_ERROR = (
    "source_a_command_available_parser_error"
)
SOURCE_A_COMMAND_AVAILABLE_INVOCATION_ERROR = (
    "source_a_command_available_invocation_error"
)
SOURCE_A_COMMAND_AVAILABLE_OUTPUT_SUPPRESSED = (
    "source_a_command_available_output_suppressed"
)
SOURCE_A_LOOPBACK_ONLY_WITH_PID_EXACT = (
    "source_a_loopback_only_with_pid_exact"
)
SOURCE_A_LOOPBACK_ONLY_WITHOUT_PID = "source_a_loopback_only_without_pid"
SOURCE_A_NON_LOOPBACK_DETECTED = "source_a_non_loopback_detected"
SOURCE_A_WILDCARD_DETECTED = "source_a_wildcard_detected"
SOURCE_A_UNKNOWN_OR_UNPARSED = "source_a_unknown_or_unparsed"
SOURCE_A_CONFLICTING = "source_a_conflicting"
SOURCE_A_RAW_OUTPUT_REQUIRED = "source_a_raw_output_required"

SOURCE_B_COMMAND_UNAVAILABLE = "source_b_command_unavailable"
SOURCE_B_COMMAND_AVAILABLE_NO_ROWS = "source_b_command_available_no_rows"
SOURCE_B_COMMAND_AVAILABLE_NO_MATCHING_ROWS = (
    "source_b_command_available_no_matching_rows"
)
SOURCE_B_COMMAND_AVAILABLE_PARSER_ERROR = (
    "source_b_command_available_parser_error"
)
SOURCE_B_COMMAND_AVAILABLE_INVOCATION_ERROR = (
    "source_b_command_available_invocation_error"
)
SOURCE_B_COMMAND_AVAILABLE_OUTPUT_SUPPRESSED = (
    "source_b_command_available_output_suppressed"
)
SOURCE_B_LOOPBACK_ONLY_WITH_PID_EXACT = (
    "source_b_loopback_only_with_pid_exact"
)
SOURCE_B_LOOPBACK_ONLY_WITHOUT_PID = "source_b_loopback_only_without_pid"
SOURCE_B_NON_LOOPBACK_DETECTED = "source_b_non_loopback_detected"
SOURCE_B_WILDCARD_DETECTED = "source_b_wildcard_detected"
SOURCE_B_UNKNOWN_OR_UNPARSED = "source_b_unknown_or_unparsed"
SOURCE_B_CONFLICTING = "source_b_conflicting"
SOURCE_B_RAW_OUTPUT_REQUIRED = "source_b_raw_output_required"

SOURCE_C_TYPE_UNAVAILABLE = "source_c_type_unavailable"
SOURCE_C_TYPE_AVAILABLE_NO_ROWS = "source_c_type_available_no_rows"
SOURCE_C_TYPE_AVAILABLE_PARSER_ERROR = "source_c_type_available_parser_error"
SOURCE_C_LOOPBACK_ONLY_WITHOUT_PID = "source_c_loopback_only_without_pid"
SOURCE_C_NON_LOOPBACK_DETECTED = "source_c_non_loopback_detected"
SOURCE_C_WILDCARD_DETECTED = "source_c_wildcard_detected"
SOURCE_C_UNKNOWN_OR_UNPARSED = "source_c_unknown_or_unparsed"
SOURCE_C_CONFLICTING = "source_c_conflicting"
SOURCE_C_RAW_OUTPUT_REQUIRED = "source_c_raw_output_required"

PROCEED_ALLOWED = "proceed_allowed"
STOP_REQUIRED = "stop_required"
SELECTED_POST_ALLOWED = "selected_post_allowed"
SELECTED_POST_NOT_ALLOWED = "selected_post_not_allowed"
ALL_SUCCESSFUL_DECISION_ROWS_PARSED = "all_successful_decision_rows_parsed"
PARSER_ISSUE_DETECTED = "parser_issue_detected"
NOT_PRINTED_SANITIZED_CATEGORIES_ONLY = (
    "not_printed_sanitized_categories_only"
)
PID_MATCH_EXACT_STARTED_PROCESS = "pid_match_exact_started_process"
PID_MATCH_CHILD_OR_RUNTIME_PROCESS = "pid_match_child_or_runtime_process"
PID_NOT_AVAILABLE = "pid_not_available"
PID_MISMATCH_OR_AMBIGUOUS = "pid_mismatch_or_ambiguous"
TWO_INDEPENDENT_SOURCES_AGREE_LOOPBACK_ONLY = (
    "two_independent_sources_agree_loopback_only"
)
NOT_VERIFIED = "not_verified"

SOURCE_A_CATEGORIES = (
    SOURCE_A_COMMAND_UNAVAILABLE,
    SOURCE_A_COMMAND_AVAILABLE_NO_ROWS,
    SOURCE_A_COMMAND_AVAILABLE_NO_MATCHING_ROWS,
    SOURCE_A_COMMAND_AVAILABLE_PARSER_ERROR,
    SOURCE_A_COMMAND_AVAILABLE_INVOCATION_ERROR,
    SOURCE_A_COMMAND_AVAILABLE_OUTPUT_SUPPRESSED,
    SOURCE_A_LOOPBACK_ONLY_WITH_PID_EXACT,
    SOURCE_A_LOOPBACK_ONLY_WITHOUT_PID,
    SOURCE_A_NON_LOOPBACK_DETECTED,
    SOURCE_A_WILDCARD_DETECTED,
    SOURCE_A_UNKNOWN_OR_UNPARSED,
    SOURCE_A_CONFLICTING,
    SOURCE_A_RAW_OUTPUT_REQUIRED,
)

SOURCE_B_CATEGORIES = (
    SOURCE_B_COMMAND_UNAVAILABLE,
    SOURCE_B_COMMAND_AVAILABLE_NO_ROWS,
    SOURCE_B_COMMAND_AVAILABLE_NO_MATCHING_ROWS,
    SOURCE_B_COMMAND_AVAILABLE_PARSER_ERROR,
    SOURCE_B_COMMAND_AVAILABLE_INVOCATION_ERROR,
    SOURCE_B_COMMAND_AVAILABLE_OUTPUT_SUPPRESSED,
    SOURCE_B_LOOPBACK_ONLY_WITH_PID_EXACT,
    SOURCE_B_LOOPBACK_ONLY_WITHOUT_PID,
    SOURCE_B_NON_LOOPBACK_DETECTED,
    SOURCE_B_WILDCARD_DETECTED,
    SOURCE_B_UNKNOWN_OR_UNPARSED,
    SOURCE_B_CONFLICTING,
    SOURCE_B_RAW_OUTPUT_REQUIRED,
)

SOURCE_C_CATEGORIES = (
    SOURCE_C_TYPE_UNAVAILABLE,
    SOURCE_C_TYPE_AVAILABLE_NO_ROWS,
    SOURCE_C_TYPE_AVAILABLE_PARSER_ERROR,
    SOURCE_C_LOOPBACK_ONLY_WITHOUT_PID,
    SOURCE_C_NON_LOOPBACK_DETECTED,
    SOURCE_C_WILDCARD_DETECTED,
    SOURCE_C_UNKNOWN_OR_UNPARSED,
    SOURCE_C_CONFLICTING,
    SOURCE_C_RAW_OUTPUT_REQUIRED,
)

LOOPBACK_SUCCESS_CATEGORIES = frozenset(
    {
        SOURCE_A_LOOPBACK_ONLY_WITH_PID_EXACT,
        SOURCE_A_LOOPBACK_ONLY_WITHOUT_PID,
        SOURCE_B_LOOPBACK_ONLY_WITH_PID_EXACT,
        SOURCE_B_LOOPBACK_ONLY_WITHOUT_PID,
        SOURCE_C_LOOPBACK_ONLY_WITHOUT_PID,
    }
)

NO_EVIDENCE_CATEGORIES = frozenset(
    {
        SOURCE_A_COMMAND_AVAILABLE_NO_ROWS,
        SOURCE_A_COMMAND_AVAILABLE_NO_MATCHING_ROWS,
        SOURCE_B_COMMAND_AVAILABLE_NO_ROWS,
        SOURCE_B_COMMAND_AVAILABLE_NO_MATCHING_ROWS,
        SOURCE_C_TYPE_AVAILABLE_NO_ROWS,
    }
)

STOP_REQUIRED_CATEGORIES = frozenset(
    {
        SOURCE_A_COMMAND_UNAVAILABLE,
        SOURCE_A_COMMAND_AVAILABLE_PARSER_ERROR,
        SOURCE_A_COMMAND_AVAILABLE_INVOCATION_ERROR,
        SOURCE_A_COMMAND_AVAILABLE_OUTPUT_SUPPRESSED,
        SOURCE_A_NON_LOOPBACK_DETECTED,
        SOURCE_A_WILDCARD_DETECTED,
        SOURCE_A_UNKNOWN_OR_UNPARSED,
        SOURCE_A_CONFLICTING,
        SOURCE_A_RAW_OUTPUT_REQUIRED,
        SOURCE_B_COMMAND_UNAVAILABLE,
        SOURCE_B_COMMAND_AVAILABLE_PARSER_ERROR,
        SOURCE_B_COMMAND_AVAILABLE_INVOCATION_ERROR,
        SOURCE_B_COMMAND_AVAILABLE_OUTPUT_SUPPRESSED,
        SOURCE_B_NON_LOOPBACK_DETECTED,
        SOURCE_B_WILDCARD_DETECTED,
        SOURCE_B_UNKNOWN_OR_UNPARSED,
        SOURCE_B_CONFLICTING,
        SOURCE_B_RAW_OUTPUT_REQUIRED,
        SOURCE_C_TYPE_UNAVAILABLE,
        SOURCE_C_TYPE_AVAILABLE_PARSER_ERROR,
        SOURCE_C_NON_LOOPBACK_DETECTED,
        SOURCE_C_WILDCARD_DETECTED,
        SOURCE_C_UNKNOWN_OR_UNPARSED,
        SOURCE_C_CONFLICTING,
        SOURCE_C_RAW_OUTPUT_REQUIRED,
    }
)

_WILDCARD_ADDRESSES = {"*", "0.0.0.0", "::", "::0", "[::]"}
_LOCALHOST_NAMES = {"localhost"}


@dataclass(frozen=True)
class _ParsedEndpoint:
    address: str | None
    port: int | None
    pid: int | None = None


def _bool_text(value: bool | None) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return NOT_VERIFIED


def _as_list(rows: Iterable[Any] | None) -> list[Any]:
    if rows is None:
        return []
    if isinstance(rows, (str, bytes)):
        return [rows]
    try:
        return list(rows)
    except TypeError:
        return [rows]


def _mapping_value(row: Any, names: tuple[str, ...]) -> Any:
    if isinstance(row, Mapping):
        lowered = {str(key).lower(): value for key, value in row.items()}
        for name in names:
            value = lowered.get(name.lower())
            if value is not None:
                return value
    for name in names:
        if hasattr(row, name):
            value = getattr(row, name)
            if value is not None:
                return value
    return None


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return None
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _strip_brackets(address: str) -> str:
    value = address.strip()
    if value.startswith("[") and value.endswith("]"):
        return value[1:-1]
    return value


def _address_kind(address: Any) -> str:
    if address is None:
        return "unknown"
    text = _strip_brackets(str(address).strip())
    if text == "":
        return "unknown"
    if text.lower() in _LOCALHOST_NAMES:
        return "loopback"
    if text in _WILDCARD_ADDRESSES:
        return "wildcard"
    try:
        parsed = ip_address(text)
    except ValueError:
        return "unknown"
    if parsed.is_unspecified:
        return "wildcard"
    if parsed.is_loopback:
        return "loopback"
    return "non_loopback"


def _parse_netstat_endpoint(endpoint: Any) -> tuple[str | None, int | None]:
    if endpoint is None:
        return None, None
    text = str(endpoint).strip()
    if text == "":
        return None, None
    if text.startswith("[") and "]:" in text:
        split_at = text.rfind("]:")
        address = text[1:split_at]
        port_text = text[split_at + 2 :]
    elif ":" in text:
        address, port_text = text.rsplit(":", 1)
    else:
        return None, None
    port = _parse_int(port_text)
    return address, port


def _parse_netstat_line(row: str) -> _ParsedEndpoint | None:
    text = row.strip()
    if not text.upper().startswith("TCP"):
        return None
    parts = [part for part in text.split() if part]
    if len(parts) < 4:
        return _ParsedEndpoint(None, None, None)
    state_index = None
    for index, part in enumerate(parts):
        if part.upper() == "LISTENING":
            state_index = index
            break
    if state_index is None:
        return None
    if len(parts) <= state_index + 1:
        return _ParsedEndpoint(None, None, None)
    address, port = _parse_netstat_endpoint(parts[1])
    pid = _parse_int(parts[state_index + 1])
    return _ParsedEndpoint(address, port, pid)


def _parse_row_endpoint(row: Any, *, source_b: bool = False) -> _ParsedEndpoint | None:
    if source_b and isinstance(row, str):
        return _parse_netstat_line(row)
    address = _mapping_value(
        row,
        ("address", "local_address", "localaddress", "LocalAddress"),
    )
    port = _mapping_value(row, ("port", "local_port", "localport", "LocalPort"))
    pid = _mapping_value(
        row,
        ("pid", "owning_process", "owningprocess", "OwningProcess"),
    )
    if address is None and port is None and pid is None:
        return _ParsedEndpoint(None, None, None)
    return _ParsedEndpoint(str(address) if address is not None else None, _parse_int(port), _parse_int(pid))


def associate_pid(
    candidate_pid: Any,
    started_pid: Any = None,
    approved_runtime_pids: Iterable[Any] | None = None,
) -> str:
    """Return a sanitized PID association category for supplied PIDs only."""

    candidate = _parse_int(candidate_pid)
    if candidate is None:
        return PID_NOT_AVAILABLE
    started = _parse_int(started_pid)
    if started is not None and candidate == started:
        return PID_MATCH_EXACT_STARTED_PROCESS
    approved = {
        parsed
        for parsed in (_parse_int(value) for value in _as_list(approved_runtime_pids))
        if parsed is not None
    }
    if candidate in approved:
        return PID_MATCH_CHILD_OR_RUNTIME_PROCESS
    return PID_MISMATCH_OR_AMBIGUOUS


def _combined_pid_association(
    pids: Iterable[int | None],
    started_pid: Any = None,
    approved_runtime_pids: Iterable[Any] | None = None,
) -> str:
    categories = [
        associate_pid(pid, started_pid, approved_runtime_pids)
        for pid in pids
        if pid is not None
    ]
    if not categories:
        return PID_NOT_AVAILABLE
    if PID_MATCH_EXACT_STARTED_PROCESS in categories:
        return PID_MATCH_EXACT_STARTED_PROCESS
    if PID_MATCH_CHILD_OR_RUNTIME_PROCESS in categories:
        return PID_MATCH_CHILD_OR_RUNTIME_PROCESS
    return PID_MISMATCH_OR_AMBIGUOUS


def _source_result(
    *,
    category: str,
    selected_port_row_count: int = 0,
    pid_association_category: str = PID_NOT_AVAILABLE,
    non_loopback_detected_boolean: str = NOT_VERIFIED,
    parser_result_category: str = NOT_VERIFIED,
) -> dict[str, Any]:
    return {
        "category": category,
        "source_category": category,
        "selected_port_row_count": selected_port_row_count,
        "source_success_boolean": _bool_text(
            category in LOOPBACK_SUCCESS_CATEGORIES
        ),
        "source_no_evidence_boolean": _bool_text(
            category in NO_EVIDENCE_CATEGORIES
        ),
        "pid_association_category": pid_association_category,
        "non_loopback_detected_boolean": non_loopback_detected_boolean,
        "parser_result_category": parser_result_category,
        "raw_output_category": NOT_PRINTED_SANITIZED_CATEGORIES_ONLY,
    }


def _classify_pid_source_rows(
    *,
    prefix: str,
    rows: Iterable[Any] | None,
    selected_port: int,
    started_pid: Any = None,
    approved_runtime_pids: Iterable[Any] | None = None,
    command_available: bool = True,
    invocation_error: bool = False,
    output_suppressed: bool = False,
    raw_output_required: bool = False,
    source_b: bool = False,
) -> dict[str, Any]:
    categories = SOURCE_B_CATEGORIES if prefix == "source_b" else SOURCE_A_CATEGORIES
    if not command_available:
        return _source_result(category=categories[0])
    if invocation_error:
        return _source_result(category=categories[4], parser_result_category=PARSER_ISSUE_DETECTED)
    if output_suppressed:
        return _source_result(category=categories[5], parser_result_category=PARSER_ISSUE_DETECTED)
    if raw_output_required:
        return _source_result(category=categories[-1], parser_result_category=PARSER_ISSUE_DETECTED)

    row_list = _as_list(rows)
    if not row_list:
        return _source_result(category=categories[1])

    parsed_rows: list[_ParsedEndpoint] = []
    parser_error = False
    for row in row_list:
        parsed = _parse_row_endpoint(row, source_b=source_b)
        if parsed is None:
            continue
        if parsed.port is None and parsed.address is None and parsed.pid is None:
            parser_error = True
            continue
        parsed_rows.append(parsed)

    if not parsed_rows and parser_error:
        return _source_result(category=categories[3], parser_result_category=PARSER_ISSUE_DETECTED)

    matching = [row for row in parsed_rows if row.port == selected_port]
    if not matching:
        return _source_result(category=categories[2])

    kinds = [_address_kind(row.address) for row in matching]
    if "unknown" in kinds:
        return _source_result(
            category=categories[10],
            selected_port_row_count=len(matching),
            parser_result_category=PARSER_ISSUE_DETECTED,
        )
    if "wildcard" in kinds:
        return _source_result(
            category=categories[9],
            selected_port_row_count=len(matching),
            non_loopback_detected_boolean="true",
            parser_result_category=ALL_SUCCESSFUL_DECISION_ROWS_PARSED,
        )
    if "non_loopback" in kinds:
        return _source_result(
            category=categories[8],
            selected_port_row_count=len(matching),
            non_loopback_detected_boolean="true",
            parser_result_category=ALL_SUCCESSFUL_DECISION_ROWS_PARSED,
        )
    if set(kinds) != {"loopback"}:
        return _source_result(
            category=categories[11],
            selected_port_row_count=len(matching),
            parser_result_category=PARSER_ISSUE_DETECTED,
        )

    pid_category = _combined_pid_association(
        (row.pid for row in matching),
        started_pid,
        approved_runtime_pids,
    )
    if pid_category in {
        PID_MATCH_EXACT_STARTED_PROCESS,
        PID_MATCH_CHILD_OR_RUNTIME_PROCESS,
    }:
        category = categories[6]
    else:
        category = categories[7]
    return _source_result(
        category=category,
        selected_port_row_count=len(matching),
        pid_association_category=pid_category,
        non_loopback_detected_boolean="false",
        parser_result_category=ALL_SUCCESSFUL_DECISION_ROWS_PARSED,
    )


def classify_source_a_rows(
    rows: Iterable[Any] | None,
    selected_port: int,
    started_pid: Any = None,
    approved_runtime_pids: Iterable[Any] | None = None,
    *,
    command_available: bool = True,
    invocation_error: bool = False,
    output_suppressed: bool = False,
    raw_output_required: bool = False,
) -> dict[str, Any]:
    """Classify Source A in-memory rows into sanitized categories only."""

    return _classify_pid_source_rows(
        prefix="source_a",
        rows=rows,
        selected_port=selected_port,
        started_pid=started_pid,
        approved_runtime_pids=approved_runtime_pids,
        command_available=command_available,
        invocation_error=invocation_error,
        output_suppressed=output_suppressed,
        raw_output_required=raw_output_required,
    )


def classify_source_b_rows(
    rows: Iterable[Any] | None,
    selected_port: int,
    started_pid: Any = None,
    approved_runtime_pids: Iterable[Any] | None = None,
    *,
    command_available: bool = True,
    invocation_error: bool = False,
    output_suppressed: bool = False,
    raw_output_required: bool = False,
) -> dict[str, Any]:
    """Classify Source B in-memory netstat-like rows into categories only."""

    return _classify_pid_source_rows(
        prefix="source_b",
        rows=rows,
        selected_port=selected_port,
        started_pid=started_pid,
        approved_runtime_pids=approved_runtime_pids,
        command_available=command_available,
        invocation_error=invocation_error,
        output_suppressed=output_suppressed,
        raw_output_required=raw_output_required,
        source_b=True,
    )


def classify_source_c_rows(
    rows: Iterable[Any] | None,
    selected_port: int,
    *,
    type_available: bool = True,
    raw_output_required: bool = False,
) -> dict[str, Any]:
    """Classify Source C in-memory listener rows into sanitized categories."""

    if not type_available:
        return _source_result(category=SOURCE_C_TYPE_UNAVAILABLE)
    if raw_output_required:
        return _source_result(
            category=SOURCE_C_RAW_OUTPUT_REQUIRED,
            parser_result_category=PARSER_ISSUE_DETECTED,
        )

    row_list = _as_list(rows)
    if not row_list:
        return _source_result(category=SOURCE_C_TYPE_AVAILABLE_NO_ROWS)

    parsed_rows: list[_ParsedEndpoint] = []
    parser_error = False
    for row in row_list:
        parsed = _parse_row_endpoint(row)
        if parsed is None:
            continue
        if parsed.port is None and parsed.address is None:
            parser_error = True
            continue
        parsed_rows.append(parsed)

    if not parsed_rows and parser_error:
        return _source_result(
            category=SOURCE_C_TYPE_AVAILABLE_PARSER_ERROR,
            parser_result_category=PARSER_ISSUE_DETECTED,
        )

    matching = [row for row in parsed_rows if row.port == selected_port]
    if not matching:
        return _source_result(category=SOURCE_C_TYPE_AVAILABLE_NO_ROWS)

    kinds = [_address_kind(row.address) for row in matching]
    if "unknown" in kinds:
        return _source_result(
            category=SOURCE_C_UNKNOWN_OR_UNPARSED,
            selected_port_row_count=len(matching),
            parser_result_category=PARSER_ISSUE_DETECTED,
        )
    if "wildcard" in kinds:
        return _source_result(
            category=SOURCE_C_WILDCARD_DETECTED,
            selected_port_row_count=len(matching),
            non_loopback_detected_boolean="true",
            parser_result_category=ALL_SUCCESSFUL_DECISION_ROWS_PARSED,
        )
    if "non_loopback" in kinds:
        return _source_result(
            category=SOURCE_C_NON_LOOPBACK_DETECTED,
            selected_port_row_count=len(matching),
            non_loopback_detected_boolean="true",
            parser_result_category=ALL_SUCCESSFUL_DECISION_ROWS_PARSED,
        )
    if set(kinds) != {"loopback"}:
        return _source_result(
            category=SOURCE_C_CONFLICTING,
            selected_port_row_count=len(matching),
            parser_result_category=PARSER_ISSUE_DETECTED,
        )
    return _source_result(
        category=SOURCE_C_LOOPBACK_ONLY_WITHOUT_PID,
        selected_port_row_count=len(matching),
        non_loopback_detected_boolean="false",
        parser_result_category=ALL_SUCCESSFUL_DECISION_ROWS_PARSED,
    )


def aggregate_listener_decision(
    source_a_category: str,
    source_b_category: str,
    source_c_category: str,
    pid_association_category: str,
) -> dict[str, Any]:
    """Aggregate sanitized source categories into request-permission fields."""

    categories = (source_a_category, source_b_category, source_c_category)
    source_success_count = sum(
        1 for category in categories if category in LOOPBACK_SUCCESS_CATEGORIES
    )
    source_no_evidence_count = sum(
        1 for category in categories if category in NO_EVIDENCE_CATEGORIES
    )
    stop_required = any(category in STOP_REQUIRED_CATEGORIES for category in categories)
    pid_allowed = pid_association_category in {
        PID_MATCH_EXACT_STARTED_PROCESS,
        PID_MATCH_CHILD_OR_RUNTIME_PROCESS,
    }
    proceed = (
        source_success_count >= 2
        and source_no_evidence_count <= 1
        and pid_allowed
        and not stop_required
    )
    non_loopback = "false" if proceed else NOT_VERIFIED
    if any("non_loopback_detected" in category for category in categories):
        non_loopback = "true"
    parser_result = (
        ALL_SUCCESSFUL_DECISION_ROWS_PARSED
        if proceed
        else PARSER_ISSUE_DETECTED
        if stop_required
        else NOT_VERIFIED
    )
    return {
        "clarified_live_listener_verification_category": (
            PROCEED_ALLOWED if proceed else STOP_REQUIRED
        ),
        "bind_verifier_decision_category": PROCEED_ALLOWED if proceed else STOP_REQUIRED,
        "request_permission_category": (
            SELECTED_POST_ALLOWED if proceed else SELECTED_POST_NOT_ALLOWED
        ),
        "source_success_count": source_success_count,
        "source_no_evidence_count": source_no_evidence_count,
        "source_agreement_category": (
            TWO_INDEPENDENT_SOURCES_AGREE_LOOPBACK_ONLY if proceed else NOT_VERIFIED
        ),
        "non_loopback_detected_boolean": non_loopback,
        "pid_association_category": pid_association_category,
        "parser_result_category": parser_result,
        "raw_output_category": NOT_PRINTED_SANITIZED_CATEGORIES_ONLY,
    }


def build_sanitized_preflight_result(
    get_net_tcp_connection_available: bool | None,
    netstat_available: bool | None,
    dotnet_ipglobalproperties_available: bool | None,
) -> dict[str, str]:
    """Build sanitized method-availability fields without running commands."""

    all_available = (
        get_net_tcp_connection_available is True
        and netstat_available is True
        and dotnet_ipglobalproperties_available is True
    )
    return {
        "get_net_tcp_connection_command_available": _bool_text(
            get_net_tcp_connection_available
        ),
        "netstat_command_available": _bool_text(netstat_available),
        "dotnet_ipglobalproperties_available": _bool_text(
            dotnet_ipglobalproperties_available
        ),
        "preflight_availability_category": (
            "all_available" if all_available else STOP_REQUIRED
        ),
    }


__all__ = [
    "SOURCE_A_CATEGORIES",
    "SOURCE_B_CATEGORIES",
    "SOURCE_C_CATEGORIES",
    "STOP_REQUIRED_CATEGORIES",
    "classify_source_a_rows",
    "classify_source_b_rows",
    "classify_source_c_rows",
    "associate_pid",
    "aggregate_listener_decision",
    "build_sanitized_preflight_result",
]
