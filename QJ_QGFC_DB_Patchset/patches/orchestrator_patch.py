# patches/orchestrator_patch.py
"""
orchestrator.py 내 수집 직후(병합 리스트 `items`)에 아래 로직을 삽입하여 QG/FC/스코어링을 적용하세요.
의존: app/qgfc.py, config.json에 qg/fc/scoring 블록 병합

예시:
    from app.qgfc import apply_quality_gates, fc_confirm, compute_score
    import json
    cfg = json.load(open("config.json","r",encoding="utf-8"))
    W = cfg["scoring"]["weights"]; TRUST = cfg["scoring"]["source_trust"]

    # 1) QG
    for it in items:
        it["qg_pass"] = apply_quality_gates(it, cfg)

    # 2) FC (제목 유사 + 시간창)
    for it in items:
        it["fc_pass"] = fc_confirm(it, items, cfg)

    # 3) 스코어
    for it in items:
        it["score"] = compute_score(it, W, TRUST)

    # 4) 스코어 순으로 TopN 자동 승인 (서버의 approve_topN과 중복 방지를 원하면 한쪽만 사용)
    TOPN = int(cfg["scoring"].get("top_n_auto_approve", 20))
    ranked = sorted([x for x in items if x.get("qg_pass")], key=lambda x: x.get("score",0), reverse=True)
    for i, it in enumerate(ranked):
        it["approved"] = (i < TOPN)
        it["state"] = "ready" if it["approved"] else it.get("state","candidate")
"""
