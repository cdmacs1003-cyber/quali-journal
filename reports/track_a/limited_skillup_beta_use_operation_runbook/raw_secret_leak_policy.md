# Raw Leak And Secret Handling Policy

R9ZDF_PACKET=T-A1-07SOU_R9ZDF_LIMITED_SKILLUP_BETA_USE_OPERATION_RUNBOOK_MATERIALIZATION_PACKET_WITH_LIMITS

## Raw Leak Boundaries

Raw text export is forbidden.
Internal path leak is forbidden.
Raw prompt output is forbidden.
Secret leak is forbidden.
Instructor-guide raw leak is forbidden.

RAW_TEXT_EXPORT_COUNT=0_CARRIED_FORWARD
INTERNAL_PATH_LEAK_COUNT=0_CARRIED_FORWARD
RAW_PROMPT_OUTPUT_COUNT=0_CARRIED_FORWARD
SECRET_LEAK_COUNT=0_CARRIED_FORWARD
INSTRUCTOR_GUIDE_RAW_LEAK_COUNT=0_CARRIED_FORWARD

RAW_LEAK_GATE=PASS_WITH_LIMITS_CARRIED_FORWARD

## Secret-Like File Handling

SECRET_LIKE_FILE_STATUS=QUARANTINE_FILENAME_LEVEL_ONLY
SECRET_CONTENT_INSPECTION=FORBIDDEN

Secret-like filenames are observed only at filename level. Contents must not be read, copied, summarized, printed, inferred, reconstructed, or used as evidence.

## Stop Conditions

Raw text leak found: stop and return REJECT or REVIEW_REQUIRED.
Internal path leak found: stop and return REJECT or REVIEW_REQUIRED.
Raw prompt output leak found: stop and return REJECT or REVIEW_REQUIRED.
Secret leak found: stop and return REJECT.
Instructor-guide raw leak found: stop and return REJECT or REVIEW_REQUIRED.
