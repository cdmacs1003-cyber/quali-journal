# NOT_EXECUTED, NOT_VERIFIED, and NOT_GRANTED Matrix

## A. NOT_EXECUTED

| Item | Status | Reason |
|---|---|---|
| Full pytest | `NOT_EXECUTED` | Not approved for R9ZIK |
| Lint | `NOT_EXECUTED` | Not approved for R9ZIK |
| Build | `NOT_EXECUTED` | Not approved for R9ZIK |
| Integration tests | `NOT_EXECUTED` | Not approved for R9ZIK |
| E2E tests | `NOT_EXECUTED` | Not approved for R9ZIK |
| Runtime/server | `NOT_EXECUTED` | Runtime/server actions forbidden |
| External HTTP/browser/healthcheck | `NOT_EXECUTED` | HTTP/browser/healthcheck actions forbidden |
| DB/network | `NOT_EXECUTED` | DB/network access forbidden |
| Deploy/release/tag/push | `NOT_EXECUTED` | Deployment and release actions forbidden |
| Production smoke | `NOT_EXECUTED` | Production behavior outside scope |
| Live beta operation | `NOT_EXECUTED` | Live operation outside scope |

## B. NOT_VERIFIED

| Item | Status | Reason |
|---|---|---|
| Full regression | `NOT_VERIFIED` | Full pytest and broader regression were not run |
| Runtime behavior | `NOT_VERIFIED` | Runtime/server was not started |
| HTTP behavior | `NOT_VERIFIED` | HTTP/browser/healthcheck requests were not sent |
| DB behavior | `NOT_VERIFIED` | DB access was not performed |
| Network behavior | `NOT_VERIFIED` | Network access was not performed |
| Production behavior | `NOT_VERIFIED` | Production smoke/live operation was not executed |
| Release behavior | `NOT_VERIFIED` | Release actions were not executed |
| Deployment behavior | `NOT_VERIFIED` | Deployment actions were not executed |
| Answer quality | `NOT_VERIFIED` | Answer-quality evaluation was not executed |
| Bridge health | `NOT_VERIFIED` | Bridge runtime/health verification was not executed |
| Skillup MVP | `NOT_VERIFIED` | Skillup MVP-level verification was not executed |

## C. NOT_GRANTED

| Claim | Status | Reason |
|---|---|---|
| Track A PASS | `NOT_GRANTED` | Static closure review with limits only |
| Beta PASS | `NOT_GRANTED` | Static closure review with limits only |
| F13 PASS | `NOT_GRANTED` | Static closure review with limits only |
| Release readiness | `NOT_GRANTED` | Release behavior was not executed or approved |
| Deployment readiness | `NOT_GRANTED` | Deployment behavior was not executed or approved |
| Production readiness | `NOT_GRANTED` | Production behavior was not executed or approved |
| Full RAW_LEAK_POLICY_BLOCK_PASS | `NOT_GRANTED` | Only selected static helper evidence accepted |
| FEEDBACK_QUEUE_PASS | `NOT_GRANTED` | Only selected static helper evidence accepted |
| BETA_RELEASE_BOARD_PASS | `NOT_GRANTED` | Only selected static contract evidence accepted |
| RAW_EXPORT_POLICY_PASS | `NOT_GRANTED` | Not executed or approved in this scope |
| HOLD_POLICY_PASS | `NOT_GRANTED` | Not executed or approved in this scope |
