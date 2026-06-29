import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import admin.server_quali as server_quali


SELECTED_PATH = "/api/f13/bridge/skillup/bridge-answer"
TOGGLE_NAME = "QJ_LOCAL_ONLY_NON_SECRET_AUTH_OVERRIDE"


class SyntheticHeaders(dict):
    def __init__(self, values):
        super().__init__((key.lower(), value) for key, value in values.items())

    def __contains__(self, key):
        return super().__contains__(key.lower())

    def get(self, key, default=None):
        return super().get(key.lower(), default)


def _request(
    *,
    method="POST",
    path=SELECTED_PATH,
    client_host="127.0.0.1",
    host="127.0.0.1",
    query_params=None,
    headers=None,
):
    request_headers = {"host": host}
    if headers:
        request_headers.update(headers)
    return SimpleNamespace(
        method=method,
        url=SimpleNamespace(path=path),
        query_params={} if query_params is None else query_params,
        headers=SyntheticHeaders(request_headers),
        client=SimpleNamespace(host=client_host) if client_host is not None else None,
    )


def _helper_allows(request):
    return server_quali._is_local_only_non_secret_f13_bridge_answer_override_request(request)


def test_default_safety_falls_back_when_toggle_is_absent(monkeypatch):
    monkeypatch.delenv(TOGGLE_NAME, raising=False)
    request = _request()
    fallback_result = object()
    calls = []

    async def synthetic_authorize(req, credentials):
        calls.append((req, credentials))
        return fallback_result

    monkeypatch.setattr(server_quali, "authorize", synthetic_authorize)
    credentials = object()

    assert _helper_allows(request) is False
    assert (
        asyncio.run(
            server_quali.authorize_f13_bridge_with_local_override(
                request,
                credentials=credentials,
            )
        )
        is fallback_result
    )
    assert calls == [(request, credentials)]


def test_wrong_toggle_value_does_not_pass(monkeypatch):
    monkeypatch.setenv(TOGGLE_NAME, "0")

    assert _helper_allows(_request()) is False


def test_positive_local_only_guard_passes(monkeypatch):
    monkeypatch.setenv(TOGGLE_NAME, "1")

    assert _helper_allows(_request()) is True


def test_wrong_path_does_not_pass(monkeypatch):
    monkeypatch.setenv(TOGGLE_NAME, "1")

    assert _helper_allows(_request(path="/api/f13/bridge/skillup/not-selected")) is False


def test_wrong_method_does_not_pass(monkeypatch):
    monkeypatch.setenv(TOGGLE_NAME, "1")

    assert _helper_allows(_request(method="GET")) is False


def test_non_loopback_client_does_not_pass(monkeypatch):
    monkeypatch.setenv(TOGGLE_NAME, "1")

    assert _helper_allows(_request(client_host="192.0.2.10")) is False


def test_non_local_host_does_not_pass(monkeypatch):
    monkeypatch.setenv(TOGGLE_NAME, "1")

    assert _helper_allows(_request(host="example.invalid")) is False


def test_query_params_do_not_pass(monkeypatch):
    monkeypatch.setenv(TOGGLE_NAME, "1")

    assert _helper_allows(_request(query_params={"q": "present"})) is False


@pytest.mark.parametrize(
    "header_name",
    ["authorization", "x-admin-token", "x-api-token", "x-api-key", "cookie"],
)
def test_auth_or_cookie_header_categories_do_not_pass(monkeypatch, header_name):
    monkeypatch.setenv(TOGGLE_NAME, "1")

    assert _helper_allows(_request(headers={header_name: "present"})) is False


def test_wrapper_positive_path_does_not_call_fallback(monkeypatch):
    monkeypatch.setenv(TOGGLE_NAME, "1")
    calls = []

    async def synthetic_authorize(req, credentials):
        calls.append((req, credentials))
        raise AssertionError("fallback should not be called")

    monkeypatch.setattr(server_quali, "authorize", synthetic_authorize)

    assert (
        asyncio.run(
            server_quali.authorize_f13_bridge_with_local_override(
                _request(),
                credentials=object(),
            )
        )
        is True
    )
    assert calls == []


def test_wrapper_fallback_path_calls_synthetic_authorize(monkeypatch):
    monkeypatch.setenv(TOGGLE_NAME, "1")
    request = _request(method="GET")
    fallback_result = object()
    calls = []

    async def synthetic_authorize(req, credentials):
        calls.append((req, credentials))
        return fallback_result

    monkeypatch.setattr(server_quali, "authorize", synthetic_authorize)
    credentials = object()

    assert _helper_allows(request) is False
    assert (
        asyncio.run(
            server_quali.authorize_f13_bridge_with_local_override(
                request,
                credentials=credentials,
            )
        )
        is fallback_result
    )
    assert calls == [(request, credentials)]


def test_selected_f13_include_dependency_scope_uses_wrapper():
    tree = ast.parse(Path(server_quali.__file__).read_text(encoding="utf-8-sig"))
    dependency_targets = []

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "include_router"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "f13_bridge_router"
        ):
            continue

        for keyword in node.keywords:
            if keyword.arg != "dependencies" or not isinstance(keyword.value, ast.List):
                continue
            for dependency in keyword.value.elts:
                if (
                    isinstance(dependency, ast.Call)
                    and isinstance(dependency.func, ast.Name)
                    and dependency.func.id == "Depends"
                    and dependency.args
                    and isinstance(dependency.args[0], ast.Name)
                ):
                    dependency_targets.append(dependency.args[0].id)

    assert dependency_targets == ["authorize_f13_bridge_with_local_override"]
