# admin/db.py
# 목적: QUALI_DB_MODE가 "cloud"일 때만 Cloud SQL Connector/SQLAlchemy를 임포트·사용
# - local/test 모드에서는 임포트조차 시도하지 않아 로컬 실행 오류를 방지
# - Cloud Run(운영)에서는 환경변수로 주입된 값으로 안전하게 연결

import os


def make_engine():
    """
    반환:
      - cloud 모드: SQLAlchemy Engine (Cloud SQL Connector + pg8000 사용)
      - 그 외(local/test): None (DB 미연결로 동작)
    """
    mode = (os.getenv("QUALI_DB_MODE") or "local").lower().strip()
    if mode != "cloud":
        # 로컬/테스트 모드에서는 DB 없이 애플리케이션 동작
        return None

    # ⬇⬇⬇ cloud 모드에서만 필요한 라이브러리 '지연 임포트'
    from google.cloud.sql.connector import Connector, IPTypes
    import sqlalchemy
    import pg8000

    # Cloud Run에 주입한 환경변수 (콘솔/CLI/Cloud Build에서 설정)
    inst = os.environ["INSTANCE_CONNECTION_NAME"]  # 예: myproj:asia-northeast3:quali-pg
    user = os.environ["DB_USER"]                   # 예: appuser
    pw   = os.environ["DB_PASS"]                   # 예: 실제 비밀번호
    name = os.environ["DB_NAME"]                   # 예: quali

    # 사설 IP 경로를 쓸 경우(선택): PRIVATE_IP=true 환경변수로 스위칭
    ip_type = IPTypes.PRIVATE if os.getenv("PRIVATE_IP") else IPTypes.PUBLIC

    # Cloud SQL Connector가 인증/네트워크 경로를 처리
    connector = Connector()

    def getconn():
        # pg8000 드라이버 사용 (순수 파이썬)
        return connector.connect(
            inst,
            "pg8000",
            user=user,
            password=pw,
            db=name,
            ip_type=ip_type,
        )

    # SQLAlchemy 엔진 생성: creator 훅으로 커넥터 연결을 주입
    return sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        pool_pre_ping=True,   # 유휴 커넥션 검증
        pool_recycle=1800,    # 30분 주기 재순환
    )
