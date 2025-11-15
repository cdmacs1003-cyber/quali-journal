# QualiJournal — QG/FC/스코어링 강화 + DB 이관 패치셋
작성: 2025-10-27 23:45:51

이 패치셋은 두 가지를 **즉시 적용**할 수 있게 제공합니다.
1) **오케스트레이터 품질게이트(QG) / 팩트체크(FC) / 스코어링 강화**
2) **JSON 파일 → 데이터베이스(Cloud SQL Postgres 또는 SQLite) 이관**

> 안전하게 진행하려면 **스테이징**에서 먼저 테스트 후 프로덕션에 적용하세요.

---

## 📦 구성
```
qj_qgfc_db_patchset/
├─ README.md                         # 이 문서
├─ config.qg.template.json           # config.json에 병합할 QG/스코어링 기본값
├─ db/
│  ├─ schema.sql                     # Postgres 기본 스키마(ENUM/JSONB 포함)
│  └─ migrations/
│     ├─ 0001_init.sql               # 테이블/ENUM 생성
│     └─ 0002_indexes.sql            # 인덱스
├─ app/
│  ├─ qj_db.py                       # SQLAlchemy ORM + 세션팩토리
│  ├─ qgfc.py                        # QG/FC/스코어링 로직 (오케스트레이터에서 import)
│  └─ import_from_json.py            # 기존 JSON을 DB로 적재
├─ patches/
│  ├─ requirements.additions.txt     # 의존성 추가 목록
│  ├─ server_quali_patch.py          # 서버 예시 패치(상태/KPI를 DB로 읽기)
│  └─ orchestrator_patch.py          # 오케스트레이터 예시 패치(QG/FC/스코어링)
└─ scripts/
   ├─ run_migration.ps1              # Postgres용 마이그레이션 실행
   └─ run_import.ps1                 # JSON → DB 적재 실행
```

---

## 🚀 적용 순서 (15~30분)
1. **의존성 추가**  
   - `pip install -r patches/requirements.additions.txt`
   - 또는 `requirements.txt`에 같은 줄을 합치고 재배포

2. **DB 준비**  
   - **Postgres** (권장): Cloud SQL 인스턴스/DB 생성 후 접속 문자열 준비  
     예시: `postgresql+psycopg2://USER:PASS@HOST:5432/qj`
   - **로컬/개발**: 환경변수 `QJ_DB_URL="sqlite:///./qj.sqlite3"` 로 시작 가능

3. **스키마 배포**  
   - Postgres: `scripts/run_migration.ps1` 실행(또는 `psql -f db/migrations/*.sql`)  
   - SQLite: `python -c "from app.qj_db import create_all; create_all()"`

4. **데이터 이관(선택)**  
   - 기존 `selected_keyword_articles.json`, `selected_articles.json`을 `app/`에 복사
   - `scripts/run_import.ps1` 실행(또는 `python app/import_from_json.py`)

5. **서버/오케스트레이터 패치**  
   - `patches/server_quali_patch.py` & `patches/orchestrator_patch.py`의 **주요 함수**를
     기존 `server_quali.py`, `orchestrator.py`에 **붙여넣기/호출**  
   - `config.json`에 `config.qg.template.json` 값을 병합(없으면 추가)

6. **배포 & 스모크**  
   - Cloud Run 최신 리비전 **100% 트래픽** 전환 후 `/health`, `/api/status` 200 확인
   - UI에서 **게이트 슬라이더** 조정 → KPI 반영 & 승인/발행 정상 동작 확인

---

## 🧩 환경변수
- `QJ_DB_URL` : SQLAlchemy 접속 문자열 (예: `postgresql+psycopg2://...` 또는 `sqlite:///./qj.sqlite3`)
- (선택) `QJ_DB_ECHO=true` : SQL 로깅

---

## 🧪 합격 기준 (DoD)
- QG 실패 항목은 자동 **candidate 제외** / `qg_pass=false` 저장
- FC 미통과(증거 불충분) 항목은 **페널티**가 적용된 점수로 남음 (`fc_pass=false`)
- 스코어 기반 **Top N 자동 승인**이 기존 UI/서버 플로우로 동작
- `/api/status` KPI가 **DB 집계**로 전환되어 UI 새로고침에 일관 반영
- JSON 파일이 없어도(또는 보조로만) 정상 발행

행운을 빕니다! 🧑‍💻
