# Warnings Register

Evidence source: accepted terminal/session evidence from T-A1-07SOU_R9ZBR_TRACK_A_RAW_LEAK_BETA_GATE_APPROVAL_PACKET

## Accepted Warning Count

WARNINGS=5

| Warning class | Count | Handling |
|---|---:|---|
| python_multipart pending deprecation warning | 1 | Preserve as warning; no code change in R9ZBT |
| Pydantic class-based config deprecation warning | 4 | Preserve as warning; no code change in R9ZBT |

## Boundary

Warnings did not fail the bounded selected R9ZBR pytest run.
R9ZBT did not rerun tests.
R9ZBT did not modify source code to address warnings.

