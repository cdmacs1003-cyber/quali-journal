# Raw Leak Beta Gate Scope

This ProofPack records accepted R9ZBR terminal/session evidence for bounded local raw leak beta gate checks. It does not expand the scope to full regression, manual runtime smoke, manual HTTP smoke, production DB verification, external network verification, release, or deployment.

## Selected Tracked Test Surfaces

| Test surface | Planning purpose |
|---|---|
| admin/tests/test_f13_runtime_guard.py | runtime guard, raw leak / policy denial behavior |
| admin/tests/test_f13_bridge_api.py | bridge API response behavior and policy blocks |
| admin/tests/test_f13_bridge_contract_regression.py | bridge contract regression and forbidden exposure checks |
| admin/tests/test_f13_bridge_check_policy_response_schema.py | check_policy response schema boundary |
| admin/tests/test_f13_bridge_evidence_response_schema.py | evidence response schema and summary-only behavior |
| admin/tests/test_f13_course_library_binding.py | course-library binding boundary relevant to evidence pointers |
| admin/tests/test_skillup_bridge_hold_feedback.py | HOLD / feedback flow boundary |
| admin/tests/test_f13_skillup_bridge_runtime_wiring.py | Skillup bridge runtime wiring boundary |

## Raw Leak Gate Focus

| Focus | Evidence state |
|---|---|
| Paid-standard raw text exposure | PASS_BOUND_SELECTED_LOCAL |
| Internal repository/local path exposure | PASS_BOUND_SELECTED_LOCAL |
| Raw prompt storage exposure | PASS_BOUND_SELECTED_LOCAL |
| Role access boundary | VERIFIED_WITH_LIMITS |
| Evidence pointer / summary-only behavior | PASS_BOUND_SELECTED_LOCAL |
| Fail-closed policy denial | PASS_BOUND_SELECTED_LOCAL |

## Exclusions

Full beta role access matrix remains NOT_VERIFIED.
Production DB behavior remains NOT_VERIFIED.
External network behavior remains NOT_EXECUTED / NOT_GRANTED.
Track A PASS and Beta PASS remain NOT_GRANTED.

