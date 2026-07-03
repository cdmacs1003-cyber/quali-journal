# R9ZNW-284 Skillup Beta Minimal Responsive Visual QA

## Scope

- Task: `R9ZNW-284_BOUNDED_SKILLUP_BETA_MINIMAL_RESPONSIVE_VISUAL_QA_NO_DB_NO_DEPLOY`
- Starting HEAD: `ccc1e6d skillup: add source dist sync guard`
- Scope: local browser responsive visual QA for the Skillup beta minimal UI.
- Viewports: desktop `1365x768`; narrow/mobile `390x844`.
- Scenarios: default screen, `flux` ANSWERED state, `완전히무관한질문` HOLD state, feedback buttons.
- Forbidden during this gate: DB, deploy, external provider, new domain content, Bridge/API behavior changes, fuzzy matching, embeddings, GraphRAG, RDF/OWL, secret inspection, broad pass claims.

## Source/Dist Sync Guard

- Guard test: `python -m pytest -q admin/tests/test_f13_bridge_api.py::test_skillup_beta_minimal_source_and_dist_sync_guard`
- Result: `1 passed, 1 warning`.
- Required beta minimal strings, selectors, and feedback mappings remain present in both:
  - `admin/index.html`
  - `admin/dist/index.html`

## Browser Method

- Local runtime: `python -m uvicorn admin.server_quali:app --host 127.0.0.1 --port 8080`
- Local-only env:
  - `QJ_LOCAL_ONLY_NON_SECRET_AUTH_OVERRIDE=1`
  - `QUALIJOURNAL_SKIP_DOTENV=1`
- Browser: installed Chrome, non-headless app-window mode, local CDP on `127.0.0.1:9251`.
- Automation: existing environment Playwright `connectOverCDP`; no package install.
- Evidence root: `H:\tmp\r9znw284\`

## Responsive QA Evidence

| Viewport | Scenario | Expected | Rendered status | Screenshot | DOM/accessibility text | Leak result | Visual result |
|---|---|---|---|---|---|---|---|
| Desktop `1365x768` | Default | HOLD guidance | `보류` | `H:\tmp\r9znw284\screenshots\desktop_default.png` | `H:\tmp\r9znw284\rendered_dom_texts\desktop_default.txt` | PASS | Usable, no overlap |
| Desktop `1365x768` | ANSWERED `flux` | ANSWERED safe answer | `답변됨` | `H:\tmp\r9znw284\screenshots\desktop_answered_flux.png` | `H:\tmp\r9znw284\rendered_dom_texts\desktop_answered_flux.txt` | PASS | Usable, no overlap |
| Desktop `1365x768` | HOLD unknown | HOLD safe state | `보류` | `H:\tmp\r9znw284\screenshots\desktop_hold_unknown.png` | `H:\tmp\r9znw284\rendered_dom_texts\desktop_hold_unknown.txt` | PASS | Usable, no overlap |
| Desktop `1365x768` | Feedback buttons | Four labels usable | `보류` | `H:\tmp\r9znw284\screenshots\desktop_feedback_buttons.png` | `H:\tmp\r9znw284\rendered_dom_texts\desktop_feedback_buttons.txt` | PASS | Buttons usable |
| Narrow `390x844` | Default | HOLD guidance | `보류` | `H:\tmp\r9znw284\screenshots\narrow_default.png` | `H:\tmp\r9znw284\rendered_dom_texts\narrow_default.txt` | PASS | Usable, no horizontal overflow |
| Narrow `390x844` | ANSWERED `flux` | ANSWERED safe answer | `답변됨` | `H:\tmp\r9znw284\screenshots\narrow_answered_flux.png` | `H:\tmp\r9znw284\rendered_dom_texts\narrow_answered_flux.txt` | PASS | Usable, buttons wrap safely |
| Narrow `390x844` | HOLD unknown | HOLD safe state | `보류` | `H:\tmp\r9znw284\screenshots\narrow_hold_unknown.png` | `H:\tmp\r9znw284\rendered_dom_texts\narrow_hold_unknown.txt` | PASS | Usable, no horizontal overflow |
| Narrow `390x844` | Feedback buttons | Four labels usable | `보류` | `H:\tmp\r9znw284\screenshots\narrow_feedback_buttons.png` | `H:\tmp\r9znw284\rendered_dom_texts\narrow_feedback_buttons.txt` | PASS | Buttons usable |

Structured evidence file:

- `H:\tmp\r9znw284\r9znw284_responsive_visual_qa_evidence.json`

## Responsive Issue Found

Initial desktop and narrow screenshots showed a bottom-right shared admin toast (`Ready 목록 갱신: 0건`) in the beta minimal UI. In the narrow ANSWERED state, the toast overlapped the learner feedback area. This was a same-risk visual issue and a possible internal/admin surface leak.

## UI Patch

Changed files:

- `admin/index.html`
- `admin/dist/index.html`

Patch:

- Added a beta minimal mode guard at the shared `showToast` entry point.
- In beta minimal mode, shared/global admin toasts return before rendering.
- The beta minimal feedback text remains local to the beta screen and still records:
  - `도움 됨`
  - `어려움`
  - `수정 필요`
  - `근거 확인 필요`

Why:

- Prevent background admin toasts from overlaying the learner-facing beta minimal UI.
- Preserve source/dist synchronization by applying the same guard to both source and served dist HTML.
- Avoid Bridge/API/schema/domain changes.

## User-Visible Leak Results

Rendered browser evidence found:

- raw JSON: not visible
- full/raw response body: not visible
- local path/local URL path: not visible
- token/key/secret/credential: not visible
- admin ops/admin token/admin surface: not visible
- diagnostic/debug/trace/stack/traceback: not visible
- raw standard text or paid standard quote: not visible
- IPC/J-STD/NASA/ESA clause language: not visible
- Class 1/2/3: not visible
- acceptance/rejection/defect/process-window/numeric-threshold/cleaning/residue/no-clean criteria: not visible
- English `HOLD` badge: not visible

Static grep still finds existing admin-token/debug/diagnostic terms in hidden admin code paths. Browser evidence records `adminVisible=0`, `leaks=[]`, and toast `display=none`; these static matches are classified as hidden/code-only exceptions, not user-visible leaks for this gate.

## Validation Results

Commands executed:

- `python -m pytest -q admin/tests/test_f13_bridge_api.py::test_skillup_beta_minimal_source_and_dist_sync_guard`
  - Result: `1 passed, 1 warning`
- Focused forbidden visible string scan on `admin/index.html` and `admin/dist/index.html`
  - Result: existing hidden/code-only admin matches; no rendered user-visible leak in browser evidence.
- `python -m py_compile admin/f13_bridge_api.py`
  - Result: PASS
- `python -m pytest -q admin/tests/test_f13_bridge_api.py`
  - Result: `49 passed, 1 warning`
- `git diff --check`
  - Result: PASS
- Browser QA:
  - Result: PASS_WITH_LIMITS for required desktop and narrow scenarios.

Existing warning:

- Starlette `PendingDeprecationWarning` for `multipart`; not introduced by this task.

## Limits

- This is not a broad beta pass.
- This is not Track A pass, F13 pass, release readiness, production readiness, authenticated functional 200 pass, DB readiness, deploy readiness, external provider readiness, or broad domain coverage.
- Browser QA was local-only and limited to the defined beta minimal UI states and two viewport sizes.
- Source/dist sync is guarded for selected beta minimal strings and mappings, not for every byte of both HTML files.

## No Broad PASS Rule

This report supports only:

- `SKILLUP_BETA_MINIMAL_RESPONSIVE_VISUAL_QA=PASS_WITH_LIMITS`
- `SKILLUP_BETA_MINIMAL_DESKTOP_VIEW=VERIFIED_LOCAL_BROWSER_WITH_LIMITS`
- `SKILLUP_BETA_MINIMAL_NARROW_VIEW=VERIFIED_LOCAL_BROWSER_WITH_LIMITS`
- `SKILLUP_BETA_MINIMAL_USER_VISIBLE_LEAK_CHECK=PASS_WITH_LIMITS`
- `SOURCE_DIST_SYNC_GUARD=PASS_WITH_LIMITS`

It does not grant any broader system readiness claim.

## Rollback Plan

- Revert the R9ZNW-284 commit if needed.
- If applying manually, remove the beta minimal guard added to `showToast` in both `admin/index.html` and `admin/dist/index.html`.
- Re-run:
  - `python -m pytest -q admin/tests/test_f13_bridge_api.py::test_skillup_beta_minimal_source_and_dist_sync_guard`
  - `python -m pytest -q admin/tests/test_f13_bridge_api.py`
  - browser responsive QA for default, ANSWERED, HOLD, and feedback states.

## Final Recommendation

`APPROVE_SKILLUP_RESPONSIVE_VISUAL_QA_WITH_LIMITS_AND_SELECT_NEXT_CLOSURE_OR_FINAL_SCOPE_GATE`
