# Participant And Role Boundary

R9ZDF_PACKET=T-A1-07SOU_R9ZDF_LIMITED_SKILLUP_BETA_USE_OPERATION_RUNBOOK_MATERIALIZATION_PACKET_WITH_LIMITS
STUDENT_INSTRUCTOR_REVIEWER_ADMIN_BOUNDARY=PRESERVED

## Allowed Participant Categories

Allowed only inside a later explicitly approved limited Skillup beta use operation packet:
- bounded student or learner participant
- instructor participant using safe instructor-facing views only
- reviewer participant using safe review metadata and feedback queues
- admin participant limited to approved observation and audit paths

## Disallowed Participant Categories

Disallowed without separate approval:
- public production user population
- product-wide beta participant cohort
- release candidate user population
- external customer operation beyond bounded Skillup beta use
- any participant requiring direct production DB access
- any participant requiring raw prompt, raw standard text, internal path, secret, or instructor-guide raw access

## Role Separation

Student, instructor, reviewer, and admin access must remain separated. Student output must not expose instructor-only guide text, admin-only metadata, reviewer-only trace details, internal paths, raw prompts, secrets, or raw standard text.

No student access to instructor raw guide is allowed.
No student/admin role leak is allowed.

ROLE_RISK_ACTION=HOLD_OR_DENIED
ROLE_ACCESS_PASS=GRANTED_BOUND_ROLE_ACCESS_MATRIX_PROOFPACK_ONLY
ROLE_ACCESS_PROOFPACK=CANONICAL_WITH_LIMITS
ROLE_ACCESS_PROOFPACK_COMMIT=CANONICAL_WITH_LIMITS
ROLE_ACCESS_SELECTED_TEST_RESULT=PASSED_WITH_LIMITS_CARRIED_FORWARD

## Stop Condition

Any role leak, role confusion, privilege leak, student/admin boundary failure, or instructor-guide raw leak stops the operation and requires REVIEW_REQUIRED or REJECT according to severity.
