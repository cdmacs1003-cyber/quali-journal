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
| Shape documentation wrapper status | `shapes/f13_bridge_runtime_contract_shape.md` | STATIC_DOC_GAP_CLOSED_WITH_LIMITS; wrapper exists and is committed at `662fc25` under `R9ZJJ_STATIC_DOC_WRAPPER_CANONICAL_WITH_LIMITS`; closed only for static wrapper documentation; runtime behavior remains NOT_VERIFIED; HTTP behavior remains NOT_VERIFIED; DB/network behavior remains NOT_VERIFIED; tests remain NOT_EXECUTED for this gap-map status packet; Track A PASS, Beta PASS, F13 PASS, release readiness, deployment readiness, production readiness, Bridge health PASS, answer quality PASS, and Skillup MVP PASS remain NOT_GRANTED; `raw_secret_leak_policy.md` remains QUARANTINE and was not opened, copied, hashed, summarized, inferred, deleted, inspected, printed, or used as a recovery source. |
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
