from __future__ import annotations

from admin.f13_library_safe_metadata_sidecar_release_board import (
    GO_HOLD_CUT_HOLD,
    NOT_GRANTED_CLAIMS,
    RELEASE_BOARD_DRAFT_FINAL_RECOMMENDATION,
    build_production_adjacent_placement_candidate_policy,
    build_sidecar_release_board_draft,
    validate_placement_candidate_policy,
    validate_sidecar_release_board_draft,
)
from admin.f13_library_safe_metadata_sidecar_registry import (
    ARTIFACT_STATE_APPROVED_SOURCE,
    ARTIFACT_STATE_CANONICAL_CANDIDATE,
    ARTIFACT_STATE_PROOFPACKED,
)


def _ok_validation(summary: str) -> dict[str, object]:
    return {"result_status": "OK", "ok": True, "summary": summary}


def _release_board() -> dict[str, object]:
    return build_sidecar_release_board_draft(
        release_board_id="release-board:R9ZNW-344:rehearsal",
        task_id="R9ZNW-344",
        source_commit="2191359",
        sidecar_id="sidecar:R9ZNW-344:rehearsal",
        sidecar_manifest_hash="a" * 64,
        sidecar_sqlite_hash="b" * 64,
        sidecar_json_hash="c" * 64,
        resolver_validation=_ok_validation("resolver accepted rehearsal manifest"),
        bridge_retrieval_validation=_ok_validation("Bridge retrieved rehearsal sidecar"),
        skillup_public_exposure_check={
            **_ok_validation("Skillup public output omitted private markers"),
            "forbidden_marker_hits": [],
        },
        rollback_plan="Preserve prior sidecar and repoint manifest only after reviewed approval.",
        refresh_plan="Explicit review refresh with new sidecar_id and hashes.",
    )


def test_candidate_policy_records_review_gated_non_mutating_placement() -> None:
    policy = build_production_adjacent_placement_candidate_policy(
        sidecar_root_candidate="H:/review-only/library-safe-metadata-sidecars",
        manifest_root_candidate="H:/review-only/library-safe-metadata-manifests",
    )

    decision = validate_placement_candidate_policy(policy)

    assert decision["result_status"] == "OK"
    assert policy["hash_validation_required"] is True
    assert policy["production_db_mutation_forbidden"] is True
    assert policy["production_raw_text_read_forbidden"] is True
    assert policy["public_pointer_exposure_forbidden"] is True
    assert policy["skillup_direct_db_access_forbidden"] is True
    assert policy["bridge_only_retrieval_required"] is True
    assert policy["production_root_write_allowed_for_rehearsal"] is False
    assert policy["live_production_adjacent_placement_granted"] is False
    assert set(policy["allowed_artifact_states"]) == {
        ARTIFACT_STATE_APPROVED_SOURCE,
        ARTIFACT_STATE_PROOFPACKED,
        ARTIFACT_STATE_CANONICAL_CANDIDATE,
    }


def test_candidate_policy_rejects_mutating_or_public_exposure_permissions() -> None:
    policy = build_production_adjacent_placement_candidate_policy(
        sidecar_root_candidate="H:/review-only/library-safe-metadata-sidecars",
        manifest_root_candidate="H:/review-only/library-safe-metadata-manifests",
    )
    policy["production_db_mutation_forbidden"] = False
    policy["production_raw_text_read_forbidden"] = False
    policy["public_pointer_exposure_forbidden"] = False
    policy["skillup_direct_db_access_forbidden"] = False
    policy["live_production_adjacent_placement_granted"] = True

    decision = validate_placement_candidate_policy(policy)

    assert decision["result_status"] == "HOLD"
    errors = " ".join(decision["errors"])
    assert "production_db_mutation_forbidden" in errors
    assert "production_raw_text_read_forbidden" in errors
    assert "public_pointer_exposure_forbidden" in errors
    assert "skillup_direct_db_access_forbidden" in errors
    assert "live_production_adjacent_placement_granted" in errors


def test_release_board_draft_preserves_not_granted_claims_and_live_hold() -> None:
    board = _release_board()

    decision = validate_sidecar_release_board_draft(board)

    assert decision["result_status"] == "OK"
    assert board["approval_required_before_live_use"] is True
    assert board["go_hold_cut_decision"] == GO_HOLD_CUT_HOLD
    assert board["final_recommendation"] == RELEASE_BOARD_DRAFT_FINAL_RECOMMENDATION
    assert set(NOT_GRANTED_CLAIMS).issubset(set(board["NOT_GRANTED claims preserved"]))


def test_release_board_draft_rejects_release_ready_or_missing_not_granted_claims() -> None:
    board = _release_board()
    board["approval_required_before_live_use"] = False
    board["go_hold_cut_decision"] = "GO"
    board["final_recommendation"] = "RELEASE_READY"
    board["NOT_GRANTED claims preserved"] = ["BETA_PASS"]

    decision = validate_sidecar_release_board_draft(board)

    assert decision["result_status"] == "HOLD"
    errors = " ".join(decision["errors"])
    assert "approval_required_before_live_use" in errors
    assert "go_hold_cut_decision" in errors
    assert "final_recommendation" in errors
    assert "NOT_GRANTED" in errors


def test_release_board_draft_rejects_skillup_public_marker_hits() -> None:
    board = _release_board()
    board["skillup_public_exposure_check"]["forbidden_marker_hits"] = ["qlib://"]

    decision = validate_sidecar_release_board_draft(board)

    assert decision["result_status"] == "HOLD"
    assert "forbidden marker hits" in " ".join(decision["errors"])
