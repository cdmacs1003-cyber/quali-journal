# R9ZNW-283 Skillup Source/Dist Sync Guard

## Scope

Bounded static/UI tooling guard for the Skillup beta minimal UI only.

Reviewed and guarded:
- `admin/index.html`
- `admin/dist/index.html`
- `admin/tests/test_f13_bridge_api.py`

No runtime, browser, HTTP, DB, deploy, external provider, fuzzy matching, embeddings, GraphRAG, RDF/OWL, or secret inspection was used.

## Source/Dist Risk Basis

`admin.server_quali` prefers `admin/dist/index.html` when it exists and falls back to `admin/index.html` only when dist is absent. R9ZNW-281 required a mirrored copy-only UI patch in both files. Future UI gates can produce misleading browser evidence if source and dist drift.

## Strings Compared

The guard checks that these beta minimal visible strings exist in both source and dist:
- 승인된 근거가 있는 질문은 짧게 답변하고, 근거가 없으면 보류합니다.
- 답변 상태
- 질문을 보내면 답변 또는 보류 상태를 확인합니다.
- 답변
- 현재 화면에 보여줄 수 있는 승인된 짧은 답변이 없습니다.
- 이 질문은 아직 승인된 근거가 없어 보류합니다. 근거가 준비되면 답변할 수 있습니다.
- 의견을 선택하면 이 화면에만 기록됩니다.
- 도움 됨
- 어려움
- 수정 필요
- 근거 확인 필요

The guard also checks these selectors and feedback mappings in both files:
- `beta-minimal-form`
- `beta-minimal-question`
- `beta-minimal-status`
- `beta-minimal-result`
- `data-beta-feedback="useful"` / `useful: "도움 됨"` / 도움 됨
- `data-beta-feedback="confusing"` / `confusing: "어려움"` / 어려움
- `data-beta-feedback="wrong"` / `wrong: "수정 필요"` / 수정 필요
- `data-beta-feedback="needs_review"` / `needs_review: "근거 확인 필요"` / 근거 확인 필요

## Sync Status

`admin/index.html` and `admin/dist/index.html` are synchronized for the checked Skillup beta minimal visible strings, selectors, and feedback mappings.

Guard added:
- `test_skillup_beta_minimal_source_and_dist_sync_guard`

## Changed Files

| File | Change | Reason |
|---|---|---|
| `admin/tests/test_f13_bridge_api.py` | Added focused source/dist sync guard constants and test | Prevent future Skillup beta minimal source/dist copy drift for checked strings and feedback labels |
| `reports/beta/R9ZNW-283_SKILLUP_SOURCE_DIST_SYNC_GUARD_20260703.md` | Added this report | Record scope, risk basis, validation, limits, and rollback plan |

No HTML source or dist patch was required.

## Static Validation Results

| Check | Result | Evidence |
|---|---|---|
| Required string/source-dist presence check | PASS_WITH_LIMITS | `python -m pytest -q admin/tests/test_f13_bridge_api.py::test_skillup_beta_minimal_source_and_dist_sync_guard` passed |
| Focused forbidden visible string scan | PASS_WITH_LIMITS | Matches were existing hidden/code/admin surfaces or guarded diagnostics markers; no new beta minimal user-visible leak was introduced |
| Python compile | PASS | `python -m py_compile admin/f13_bridge_api.py` |
| Selected pytest | PASS_WITH_LIMITS | `python -m pytest -q admin/tests/test_f13_bridge_api.py` -> 49 passed, 1 warning |

## Test Results

Executed:
- `python -m pytest -q admin/tests/test_f13_bridge_api.py::test_skillup_beta_minimal_source_and_dist_sync_guard`
- `python -m py_compile admin/f13_bridge_api.py`
- `python -m pytest -q admin/tests/test_f13_bridge_api.py`

Result:
- Focused sync guard: 1 passed, 1 warning
- Py compile: passed
- Selected bridge tests: 49 passed, 1 warning

Warning:
- Existing `PendingDeprecationWarning` from `starlette.formparsers` importing `multipart`; not introduced by this gate.

## Limits

- This guard only covers the listed beta minimal UI strings, selectors, and feedback mappings.
- It does not prove full source/dist semantic equivalence.
- It does not run runtime/browser review.
- It does not change Bridge behavior, domain content, evidence seeds, ontology, term registries, DB, deploy, or provider configuration.

## No Broad PASS Rule

This report does not grant broad beta, Track A, F13, release, production, authenticated functional 200, DB, deploy, external provider, or broad domain coverage readiness.

## Rollback Plan

If rollback is required, revert only:
- `admin/tests/test_f13_bridge_api.py`
- `reports/beta/R9ZNW-283_SKILLUP_SOURCE_DIST_SYNC_GUARD_20260703.md`

No runtime state, browser state, DB state, deploy state, evidence seed, ontology, term registry, or Bridge behavior cleanup is required.
