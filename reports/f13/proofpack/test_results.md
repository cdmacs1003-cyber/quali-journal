# F13 Selected Test Evidence

Task source: R9ZAT bounded local runtime verification.

## Command Summary

Approved selected test scope:

```text
python -m pytest -q admin/tests/test_f13_bridge_api.py admin/tests/test_f13_runtime_guard.py admin/tests/test_f13_bridge_contract_regression.py admin/tests/test_f13_bridge_evidence_response_schema.py --basetemp <temp path outside repo> -p no:cacheprovider
```

## Accepted Result

| Item | Evidence |
|---|---|
| Result | 58 passed, 5 warnings |
| Warning class | Starlette/Pydantic dependency deprecation warnings |
| Scope | Selected F13/Bridge/runtime guard tests only |
| Repository artifacts | None reported |
| Worktree after verification | Clean |

## Boundary

- This selected test evidence is bounded selected-scope evidence only.
- This is not a full product test suite.
- This is not release evidence by itself.
- This does not grant F13_PASS, TRACK_A_PASS, BETA_PASS, or release approval.
- No tests were rerun under R9ZAX.

