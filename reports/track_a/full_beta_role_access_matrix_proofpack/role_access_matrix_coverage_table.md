# Full Beta Role Access Matrix Coverage Table

| Matrix row | State | Evidence / limit |
|---|---|---|
| missing/unknown role fail closed | ENFORCED_WITH_EVIDENCE | R9ZCL selected role matrix evidence carried forward; R9ZCM source/test/schema commit canonical with limits. |
| missing binding/course scope | ENFORCED_WITH_EVIDENCE | R9ZCL selected role matrix evidence carried forward; protected answer flow requires binding/course scope. |
| evidence_depth by role | ENFORCED_WITH_EVIDENCE | R9ZCL selected coverage mapped student_safe, instructor_safe, review_trace_safe_metadata, and audit_trace_safe_metadata. |
| student safe summary | ENFORCED_WITH_EVIDENCE | R9ZCL selected tests carried forward student-safe summary behavior. |
| student raw standard text export block | BLOCKED_WITH_EVIDENCE | R9ZCL selected tests carried forward raw export block. |
| student internal path/raw prompt/secret leak block | BLOCKED_WITH_EVIDENCE | Zero leak counters carried forward from R9ZCL selected tests. |
| instructor safe view | ENFORCED_WITH_EVIDENCE | R9ZCL selected role matrix evidence carried forward. |
| instructor raw standard text export block | BLOCKED_WITH_EVIDENCE | R9ZCL selected role matrix evidence carried forward. |
| instructor guide raw export block | BLOCKED_WITH_EVIDENCE | R9ZCL selected role matrix evidence carried forward. |
| reviewer review_trace visibility | ENFORCED_WITH_EVIDENCE | R9ZCL selected role matrix evidence carried forward for safe review_trace metadata. |
| reviewer raw export block | BLOCKED_WITH_EVIDENCE | R9ZCL selected role matrix evidence carried forward. |
| admin safe metadata | ENFORCED_WITH_EVIDENCE | R9ZCL selected role matrix evidence carried forward for safe admin metadata. |
| admin secret/internal path/raw prompt leak block | BLOCKED_WITH_EVIDENCE | Zero leak counters carried forward from R9ZCL selected tests. |
| audit_trace visibility | ENFORCED_WITH_EVIDENCE | R9ZCL selected role matrix evidence carried forward for safe audit_trace metadata. |
| license entitlement | ENFORCED_WITH_EVIDENCE | R9ZCL selected role matrix evidence carried forward; pointer-only and HOLD behavior covered with limits. |
| tenant/org/cohort scope | ENFORCED_WITH_EVIDENCE | R9ZCL selected role matrix evidence carried forward. |
| cross-tenant denial | DENIED_WITH_EVIDENCE | R9ZCL selected role matrix evidence carried forward. |
| raw_text_export_count | CARRIED_FORWARD | RAW_TEXT_EXPORT_COUNT=0_CARRIED_FORWARD. |
| internal_path_leak_count | CARRIED_FORWARD | INTERNAL_PATH_LEAK_COUNT=0_CARRIED_FORWARD. |
| raw_prompt_output_count | CARRIED_FORWARD | RAW_PROMPT_OUTPUT_COUNT=0_CARRIED_FORWARD. |
| secret_leak_count | CARRIED_FORWARD | SECRET_LEAK_COUNT=0_CARRIED_FORWARD. |
| instructor_guide_raw_leak_count | CARRIED_FORWARD | INSTRUCTOR_GUIDE_RAW_LEAK_COUNT=0_CARRIED_FORWARD. |
| final role access pass status | NOT_GRANTED | ROLE_ACCESS_PASS=NOT_GRANTED; no Track A, Beta, Release, Product, or deployment PASS granted. |
