# F13 Local Runtime Smoke Evidence

Task source: R9ZAT bounded local runtime verification and R9ZAU boundary review.

## Accepted Local Smoke

| Check | Accepted evidence |
|---|---|
| Bind scope | 127.0.0.1 only |
| Local port | 8765 |
| Server start | Passed |
| `GET /health` | HTTP 200 |
| `GET /readyz` | HTTP 200 |
| `POST /api/f13/bridge/retrieve-evidence` | HTTP 200 |
| Bounded response summary | `result_status=OK`, `raw_text_included=False`, `internal_path_included=False`, `evidence_count=1` |
| Server stop | Confirmed stopped |
| Repository artifacts | None reported |

## Boundary

- Localhost-only scope.
- Not production runtime evidence.
- Not production DB evidence.
- Not external network verification.
- Does not grant F13_PASS, TRACK_A_PASS, BETA_PASS, product pass, release pass, tag, push, deploy, or release approval.
- No server was started and no HTTP request was sent under R9ZAX.

