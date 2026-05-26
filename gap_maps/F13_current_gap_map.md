# F13 Current Gap Map

## Static Recovery State

This gap map records the clean-worktree Bridge/F13 source recovery baseline after governance canonicalization, reports evidence sealing, and 07SQ source-surface recovery.

## P0 Gaps

| Gap | Target surface | Required closure |
|---|---|---|
| Bridge API route surface recovered | `admin/f13_bridge_api.py` | STATIC_RECOVERY_COMPLETE; runtime behavior remains NOT_VERIFIED. |
| Runtime guard regenerated | `admin/f13_runtime_guard.py` | STATIC_RECOVERY_COMPLETE; runtime behavior remains NOT_VERIFIED. |
| retrieve-evidence schema regenerated | `schemas/f13_bridge_evidence_response.schema.json` | STATIC_RECOVERY_COMPLETE; alias policy still requires static coverage. |

## P1 Gaps

| Gap | Target surface | Required closure |
|---|---|---|
| check-policy schema static coverage missing | `admin/tests/test_f13_bridge_check_policy_response_schema.py` | Add static schema test; defer execution to a later test gate. |
| explain-trace feedback-candidate coverage missing | `admin/tests/test_f13_bridge_explain_trace_response_schema.py` | Add static schema and feedback-candidate tests; defer execution to a later test gate. |
| Shape documentation wrapper missing | `shapes/f13_bridge_runtime_contract_shape.md` | Add wrapper documenting JSON shape canonicality. |
| Bridge tests present but not executed | `admin/tests/test_f13_bridge_*.py` | Defer execution to a later explicit test gate. |

## Deferred Gaps

| Gap | Status |
|---|---|
| F13 reports path | DEFERRED |
| F13 docs path | DEFERRED |
| Bridge functional 200 behavior | NOT_VERIFIED |
| Track A/Beta/F13/release approval | NOT_GRANTED |

## Required Controls

- No old dirty worktree wholesale copy.
- No secret-bearing file recovery.
- No runtime, server, HTTP, or test execution in static recovery gates.
- No readiness or approval escalation from static recovery alone.
