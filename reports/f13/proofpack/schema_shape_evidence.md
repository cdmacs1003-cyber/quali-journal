# F13 Schema and Shape Evidence

Task: R9ZAX bounded F13 ProofPack materialization.

## Static Tracked Evidence

| Area | Paths | State |
|---|---|---|
| F13 Bridge API | `admin/f13_bridge_api.py` | Present and tracked |
| F13 runtime guard | `admin/f13_runtime_guard.py` | Present and tracked |
| Skillup bridge wiring | `admin/f13_skillup_bridge.py` | Present and tracked |
| Course/library binding | `admin/f13_course_library_binding.py` | Present and tracked |
| ProofPack/gap map helpers | `admin/f13_proofpack_manifest.py`, `admin/f13_gap_map.py` | Present and tracked |
| Beta release board helper | `admin/f13_beta_release_board.py` | Present and tracked |
| Bridge schemas | `schemas/f13_bridge_evidence_response.schema.json`, `schemas/f13_bridge_check_policy_response.schema.json`, `schemas/f13_bridge_explain_trace_response.schema.json`, `schemas/f13/bridge_evidence_response.schema.json` | Present and tracked |
| Runtime contract shape | `shapes/f13_bridge_runtime_contract_shape.json`, `shapes/f13_bridge_runtime_contract_shape.md` | Present and tracked |
| Selected contract tests | `admin/tests/test_f13_bridge_contract_regression.py`, `admin/tests/test_f13_bridge_evidence_response_schema.py` | Present and tracked |

## Limitation

- This packet records static tracked evidence only.
- No tests were rerun under R9ZAX.
- No server was started under R9ZAX.
- No schema or shape hash manifest was finalized under R9ZAX.
- Final ProofPack hash and manifest verification requires a separate approval packet.

