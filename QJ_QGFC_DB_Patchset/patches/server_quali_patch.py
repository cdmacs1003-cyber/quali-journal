# patches/server_quali_patch.py
"""
server_quali.py에 아래 함수를 추가/호출하면 KPI·게이트 처리 등을 DB로 전환할 수 있습니다.
- 의존: app/qj_db.py (SQLAlchemy), 환경변수 QJ_DB_URL
1) 파일 상단에 import 추가:
    from app.qj_db import get_session, get_or_create_edition, kpi_for_edition, approve_top_n
2) /api/status 핸들러 내부:
    sess = get_session()
    ed = get_or_create_edition(sess, etype, edate, keyword)  # etype='keyword'|'daily' 등
    return JSONResponse(kpi_for_edition(sess, ed))
3) /api/config/gate_required PATCH:
    ed.gate_required = body.value; sess.commit()
4) 자동 승인 API(또는 버튼 처리 시):
    approve_top_n(sess, ed, n=body.n or 20); sess.commit()
"""
