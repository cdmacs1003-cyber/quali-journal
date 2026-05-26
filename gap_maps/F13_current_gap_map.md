# F13 Current Gap Map

## Static Recovery State

This gap map records the clean-worktree Bridge/F13 source recovery baseline after governance canonicalization and reports evidence sealing.

## P0 Gaps

| Gap | Target surface | Required closure |
|---|---|---|
| Bridge API route surface missing | `admin/f13_bridge_api.py` | Recover selected source and verify route symbols statically. |
| Runtime guard missing | `admin/f13_runtime_guard.py` | Regenerate from contract with fail-closed marker handling. |
| retrieve-evidence schema missing | `schemas/f13_bridge_evidence_response.schema.json` | Regenerate JSON schema and verify parse. |

## P1 Gaps

| Gap | Target surface | Required closure |
|---|---|---|
| check-policy schema missing | `schemas/f13_bridge_check_policy_response.schema.json` | Regenerate dedicated response schema. |
| explain-trace schema missing | `schemas/f13_bridge_explain_trace_response.schema.json` | Regenerate dedicated response schema. |
| Bridge tests missing | `admin/tests/test_f13_bridge_*.py` | Recover selected tests and defer execution to a later test gate. |

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
