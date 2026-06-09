# Raw Leak Safe Output Contract

Packet: T-A1-07SOU_R9ZBY_TRACK_A_SAMPLE_QA_MATERIALIZATION_APPROVAL_PACKET

## Purpose

Define the safe output boundary for future sample QA and E2E answer evidence.

## Required Safe Output Rules

Future learner-facing output must:

- use safe_summary for answer text
- include evidence_id for ANSWERED cases
- include bridge_trace_id for traceable answer cases
- preserve raw_text_policy=SUMMARY_ONLY or stricter
- avoid raw source text
- avoid raw prompt echo
- avoid raw query echo
- avoid raw answer echo
- avoid internal repository path or local file path
- avoid secret-like content
- avoid role-inappropriate detail

## Required Negative Checks

Future proof must check that output does not contain:

| Surface | Expected |
|---|---|
| raw_text | absent |
| raw_prompt | absent |
| raw_query | absent |
| raw_answer | absent |
| internal_path | absent |
| local_route | absent |
| secret-like content | absent |
| paid standard raw text | absent |
| Track A PASS escalation | absent or NOT_GRANTED |
| Beta PASS escalation | absent or NOT_GRANTED |

## Canonical Boundary Inputs

| Input | State |
|---|---|
| Raw Leak Beta Gate ProofPack | CANONICAL |
| R9ZBR raw leak result | VERIFIED_WITH_LIMITS |
| RAW_LEAK_BETA_GATE | PASS_BOUND_SELECTED_LOCAL_WITH_ROLE_MATRIX_LIMIT |
| FULL_BETA_ROLE_ACCESS_MATRIX | NOT_VERIFIED |

## Non-Execution

RAW_LEAK_OUTPUT_EXECUTION=NOT_EXECUTED
FULL_BETA_ROLE_ACCESS_MATRIX=NOT_VERIFIED
TRACK_A_PASS=NOT_GRANTED
BETA_PASS=NOT_GRANTED
