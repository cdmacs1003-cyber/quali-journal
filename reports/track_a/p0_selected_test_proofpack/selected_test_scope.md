# Track A P0 Selected Test Scope

This ProofPack records accepted R9ZBK terminal/session evidence for selected Track A P0 pytest execution. It does not expand the scope to full regression, runtime smoke, HTTP smoke, production DB verification, external network verification, release, or deployment.

## P0 Axis Coverage

| P0 item | Selected tracked test surfaces | Evidence status |
|---|---|---|
| Bridge Runtime MVP | admin/tests/test_f13_bridge_api.py; admin/tests/test_f13_runtime_guard.py; admin/tests/test_f13_bridge_contract_regression.py; admin/tests/test_f13_bridge_evidence_response_schema.py; admin/tests/test_f13_skillup_bridge_runtime_wiring.py | EXECUTED_PASS_IN_R9ZBK |
| Skillup answer/HOLD flow | admin/tests/test_f13_skillup_bridge_runtime_wiring.py; admin/tests/test_skillup_bridge_hold_feedback.py | EXECUTED_PASS_IN_R9ZBK |
| course_library_binding | admin/tests/test_f13_course_library_binding.py | EXECUTED_PASS_IN_R9ZBK |
| raw leak / policy block | admin/tests/test_f13_runtime_guard.py; admin/tests/test_f13_bridge_api.py; admin/tests/test_f13_bridge_contract_regression.py | EXECUTED_PASS_IN_R9ZBK |
| feedback queue | admin/tests/test_skillup_bridge_hold_feedback.py | EXECUTED_PASS_IN_R9ZBK |
| Beta Release Board | admin/tests/test_f13_beta_release_board.py | EXECUTED_PASS_IN_R9ZBK |

## Selected Test File List

- admin/tests/test_f13_bridge_api.py
- admin/tests/test_f13_runtime_guard.py
- admin/tests/test_f13_bridge_contract_regression.py
- admin/tests/test_f13_bridge_evidence_response_schema.py
- admin/tests/test_f13_skillup_bridge_runtime_wiring.py
- admin/tests/test_skillup_bridge_hold_feedback.py
- admin/tests/test_f13_course_library_binding.py
- admin/tests/test_f13_beta_release_board.py

## Scope Boundary

This selected test scope supports only the statement that the accepted R9ZBK selected pytest run passed 76 tests with 5 warnings. It does not grant Track A PASS, Beta PASS, Release PASS, Product PASS, tag, push, deployment, production DB verification, or external network verification.
