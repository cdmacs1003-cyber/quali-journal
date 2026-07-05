# Owner Responsibility Approval Record Contract

Task scope: R9ZNW-369.

This contract defines the pointer-only approval-record template and review-board boundary for DOCTOR_YOON owner responsibility decisions. It is a governance and proofpack contract only. It does not execute an approval record by itself.

This contract does not approve raw text, production DB retrieval, production Library root use, durable production feedback storage, deploy, release readiness, production readiness, Beta, Track A, or F13.

## Authority Constants

```text
OWNER_RESPONSIBILITY_AUTHORITY=DOCTOR_YOON
LEGAL_DOMAIN_SEMANTIC_FINAL_RESPONSIBILITY=DOCTOR_YOON
EXTERNAL_APPROVER_DISCOVERY_LOOP=FORBIDDEN
APPROVAL_RECORD_PURPOSE=CAPTURE_OWNER_ASSUMPTION_OF_RESPONSIBILITY_AND_DECISION_BASIS
```

External legal or domain approver discovery is out of scope for this contract. The later approval-execution gate must capture whether DOCTOR_YOON assumes legal, domain, and semantic responsibility for the targeted records and evidence pointers.

## Decision Vocabulary

Approval-record fields that need a decision must use one of these values unless a narrower downstream contract defines a stricter set:

```text
APPROVED_WITH_LIMITS
HOLD
NOT_VERIFIED
REJECTED
REVIEW_REQUIRED
```

The default value for an incomplete approval record is `HOLD` or `NOT_VERIFIED`. A template-only record is never sufficient to set legal, domain, or semantic approval to verified.

## Required Approval Record Fields

### A. Identity

Each owner responsibility approval record must include:

| Field | Requirement |
|---|---|
| `approval_record_id` | Stable identifier for the approval record. |
| `approval_record_type` | Must identify rights/license/legal, semantic/domain, or combined owner responsibility approval. |
| `target_record_id` | Identifier of the seed, sidecar row, summary, policy item, or evidence object under review. |
| `target_record_type` | Type of reviewed object. |
| `standard_node_id` or `source_node_id` | Pointer-only source node identifier when available. |
| `evidence_pointer_id` | Pointer to safe metadata evidence. |
| `source_span_pointer` | Pointer-only span reference. Raw source text must not be copied into this field. |
| `created_at` | Record creation timestamp. |
| `owner_responsibility_authority` | Must be `DOCTOR_YOON`. |
| `reviewer_role` | Placeholder or role of the reviewing owner. |
| `reviewer_name_or_id` | Placeholder or owner identifier. |

### B. Owner Responsibility Declaration

Each owner responsibility approval record must include:

| Field | Requirement |
|---|---|
| `owner_responsibility_authority` | Must be `DOCTOR_YOON`. |
| `owner_assumes_legal_responsibility` | Boolean or explicit decision value. |
| `owner_assumes_domain_responsibility` | Boolean or explicit decision value. |
| `owner_assumes_semantic_responsibility` | Boolean or explicit decision value. |
| `owner_decision_basis` | Pointer-only basis, safe summary basis, or proofpack basis. No raw text. |
| `owner_decision_scope` | Exact bounded scope of the decision. |
| `owner_decision_limitations` | Limits, exclusions, non-grants, and required follow-up. |
| `owner_signature_or_approval_marker` | Placeholder for later owner approval marker. |
| `approval_effective_date` | Effective date if approved. |
| `review_cycle_or_expiration` | Review cycle or expiration condition. |
| `required_followup` | Required follow-up gate or action. |

### C. Rights, License, And Legal Review

Each rights/license/legal review record must include:

| Field | Requirement |
|---|---|
| `rights_status` | Rights status for the target record. |
| `license_status` | License status for the target record. |
| `legal_review_status` | Legal review status. |
| `raw_text_use_allowed` | Defaults to false unless a separate approved raw-text gate exists. |
| `raw_text_reproduction_allowed` | Defaults to false. Raw text reproduction remains prohibited by this contract. |
| `production_library_root_allowed` | Defaults to false unless separately approved. |
| `production_db_retrieval_allowed` | Defaults to false unless separately approved. |
| `durable_feedback_storage_allowed` | Defaults to false unless separately approved. |
| `approval_decision` | One allowed decision vocabulary value. |
| `decision_reason` | Pointer-only reason and safe summary basis. |
| `evidence_pointer_list` | List of safe evidence pointers or proofpack references. |
| `restrictions` | Restrictions that survive approval. |
| `expiration_or_review_cycle` | Expiration or required review cycle. |
| `required_followup` | Next action if approval is absent, limited, or rejected. |

### D. Semantic And Domain Review

Each semantic/domain review record must include:

| Field | Requirement |
|---|---|
| `semantic_review_status` | Semantic review status. |
| `domain_reviewer_role` | Owner, domain reviewer role, or placeholder. |
| `safe_summary_verified` | Whether the safe summary is verified within the stated scope. |
| `semantic_summary_verified` | Whether the semantic summary is verified within the stated scope. |
| `source_alignment_verified` | Whether pointer-only source alignment is verified. |
| `class_or_scope_applicability_verified_if_relevant` | Whether class/scope applicability is verified when applicable. |
| `overclaim_risk` | Low, medium, high, or explicit narrative. |
| `required_hold_conditions` | Conditions that require HOLD or NOT_VERIFIED. |
| `approval_decision` | One allowed decision vocabulary value. |
| `decision_reason` | Pointer-only reason and safe summary basis. |
| `evidence_pointer_list` | List of safe evidence pointers or proofpack references. |
| `required_followup` | Next action if approval is absent, limited, or rejected. |

### E. Claim Boundary

Each approval record must include explicit non-grant fields:

| Field | Required Value Unless Separately Approved |
|---|---|
| `does_not_grant_raw_text_approval` | true |
| `does_not_grant_production_db_approval` | true |
| `does_not_grant_production_library_root_approval` | true |
| `does_not_grant_release_ready` | true |
| `does_not_grant_deploy_ready` | true |
| `does_not_grant_production_ready` | true |
| `does_not_grant_beta_pass` | true |
| `does_not_grant_track_a_pass` | true |
| `does_not_grant_f13_pass` | true |

## Review Board

### Rights, License, And Legal Items

DOCTOR_YOON must review:

| Review Item | Required Basis |
|---|---|
| Rights metadata | Tracked safe metadata field, seed metadata, sidecar metadata, or proofpack reference. |
| License status | Tracked metadata or pointer-only evidence. |
| Legal review status | Executed owner approval record, not a seed approval id alone. |
| Raw text handling | Confirmation that raw paid standard text is not read, copied, reproduced, or emitted. |
| Production adjacency | Explicit non-grant unless a separate production DB/root gate exists. |
| Restrictions | Bounded restrictions that survive approval. |
| Follow-up | Next gate for unresolved or limited items. |

Sufficient evidence for a bounded owner decision can include safe metadata, safe summaries, evidence pointer identifiers, source span pointers, R360-R369 proofpack references, and an explicit DOCTOR_YOON responsibility declaration.

Insufficient evidence includes candidate-only docs, missing owner approval marker, missing decision basis, unsupported scope expansion, required raw text access, required production DB/root access, or an attempted external approver discovery loop.

### Semantic And Domain Items

DOCTOR_YOON must review:

| Review Item | Required Basis |
|---|---|
| Safe summary | Safe summary field and evidence pointer basis. |
| Semantic summary | Semantic summary field, if present, and source alignment pointer. |
| Source alignment | Pointer-only span or node reference. No raw text copy. |
| Applicability | Class, scope, or target applicability when relevant. |
| Overclaim risk | Explicit risk assessment and required HOLD conditions. |
| Output boundary | Confirmation that Bridge and Skillup output remains safe summary or HOLD only. |
| Follow-up | Next gate for unresolved or limited items. |

Semantic approval remains `NOT_VERIFIED` if there is no completed owner/domain semantic approval record. Local selected validation, seed approval ids, and safe metadata projection do not by themselves verify semantic correctness beyond safe summaries.

## HOLD And NOT_VERIFIED Conditions

A target must remain `HOLD`, `NOT_VERIFIED`, or `REVIEW_REQUIRED` when any of these conditions apply:

| Condition | Required Outcome |
|---|---|
| Owner approval marker is absent. | `NOT_VERIFIED` |
| Decision basis is missing or not pointer-only. | `HOLD` |
| Raw text access is required for the requested decision. | `REVIEW_REQUIRED` |
| Production DB or production Library root access is required. | `REVIEW_REQUIRED` |
| Candidate docs are the only source. | `NOT_VERIFIED` |
| Scope would grant release, deployment, production, Beta, Track A, or F13 claims. | `HOLD` |
| External approver discovery is attempted. | `HOLD` |

## Pointer-Only Empty Template

The following template is an empty record shape. It is not an approval.

```yaml
approval_record_id: ""
approval_record_type: "owner_responsibility_rights_semantic"
target_record_id: ""
target_record_type: ""
standard_node_id: ""
source_node_id: ""
evidence_pointer_id: ""
source_span_pointer: ""
created_at: ""
owner_responsibility_authority: "DOCTOR_YOON"
reviewer_role: ""
reviewer_name_or_id: ""

owner_assumes_legal_responsibility: false
owner_assumes_domain_responsibility: false
owner_assumes_semantic_responsibility: false
owner_decision_basis: ""
owner_decision_scope: ""
owner_decision_limitations: ""
owner_signature_or_approval_marker: ""
approval_effective_date: ""
review_cycle_or_expiration: ""
required_followup: ""

rights_status: "NOT_VERIFIED"
license_status: "NOT_VERIFIED"
legal_review_status: "NOT_VERIFIED"
raw_text_use_allowed: false
raw_text_reproduction_allowed: false
production_library_root_allowed: false
production_db_retrieval_allowed: false
durable_feedback_storage_allowed: false
rights_license_legal_approval_decision: "HOLD"
rights_license_legal_decision_reason: ""
rights_license_legal_evidence_pointer_list: []
rights_license_legal_restrictions: []
rights_license_legal_expiration_or_review_cycle: ""
rights_license_legal_required_followup: ""

semantic_review_status: "NOT_VERIFIED"
domain_reviewer_role: ""
safe_summary_verified: false
semantic_summary_verified: false
source_alignment_verified: false
class_or_scope_applicability_verified_if_relevant: false
overclaim_risk: "NOT_VERIFIED"
required_hold_conditions: []
semantic_domain_approval_decision: "HOLD"
semantic_domain_decision_reason: ""
semantic_domain_evidence_pointer_list: []
semantic_domain_required_followup: ""

does_not_grant_raw_text_approval: true
does_not_grant_production_db_approval: true
does_not_grant_production_library_root_approval: true
does_not_grant_release_ready: true
does_not_grant_deploy_ready: true
does_not_grant_production_ready: true
does_not_grant_beta_pass: true
does_not_grant_track_a_pass: true
does_not_grant_f13_pass: true
```

## Preserved Claim Boundary

```text
OWNER_RESPONSIBILITY_AUTHORITY=DOCTOR_YOON
LEGAL_DOMAIN_SEMANTIC_FINAL_RESPONSIBILITY=DOCTOR_YOON
EXTERNAL_APPROVER_DISCOVERY_LOOP=FORBIDDEN
APPROVAL_RECORD_PURPOSE=CAPTURE_OWNER_ASSUMPTION_OF_RESPONSIBILITY_AND_DECISION_BASIS

LOCAL_BETA_CHAIN_SELECTED_LOCAL_VALIDATION_CLOSED_WITH_LIMITS=SUPPORTED
LOCAL_ONLY_SAFE_SIDECAR_SEED_BRIDGE_SKILLUP_FEEDBACK_CHAIN_CLOSED_WITH_LIMITS=SUPPORTED

PRODUCTION_DB_OK_RETRIEVAL_PASS=NOT_GRANTED
PRODUCTION_LIBRARY_ROOT_PASS=NOT_GRANTED
RELEASE_READY=NOT_GRANTED
DEPLOYMENT_READY=NOT_GRANTED
PRODUCTION_READY=NOT_GRANTED
BETA_PASS=NOT_GRANTED
TRACK_A_PASS=NOT_GRANTED
F13_PASS=NOT_GRANTED
RAW_TEXT_APPROVAL=NOT_GRANTED
DEPLOY_APPROVAL=NOT_GRANTED
DURABLE_FEEDBACK_STORAGE_APPROVAL=NOT_GRANTED

RIGHTS_LICENSE_LEGAL_APPROVAL=NOT_VERIFIED_UNLESS_COMPLETED_OWNER_APPROVAL_RECORD_EXISTS
SEMANTIC_CORRECTNESS_BEYOND_SAFE_SUMMARIES=NOT_VERIFIED_UNLESS_COMPLETED_OWNER_DOMAIN_SEMANTIC_APPROVAL_RECORD_EXISTS
```

## Next-Gate Consumption

A later owner approval execution gate may consume this contract only by producing completed approval records with DOCTOR_YOON as the owner responsibility authority. The later gate must preserve this contract's non-grants unless it has separate explicit authority for a higher-risk boundary.

The recommended next gate after introducing this contract is:

```text
R9ZNW-370_BOUNDED_APPROVAL_TEMPLATE_POST_COMMIT_VALIDATION_AND_OWNER_REVIEW_PACKET_NO_RAW_TEXT_NO_DEPLOY
```
