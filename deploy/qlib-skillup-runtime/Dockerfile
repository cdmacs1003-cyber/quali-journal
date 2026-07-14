FROM python:3.11-slim@sha256:00af38ae2ed311628970782e8a2d7f014d8909dbc63cb97bc0a158187f4db045 AS dependencies

ARG SOURCE_DATE_EPOCH

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY deploy/qlib-skillup-runtime/requirements.lock /tmp/requirements.lock
RUN test -n "${SOURCE_DATE_EPOCH}" \
    && python -m pip install --no-cache-dir --no-compile --require-hashes \
       --target /opt/site-packages -r /tmp/requirements.lock \
    && find /opt/site-packages -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} + \
    && tar --sort=name --format=posix --mtime="@${SOURCE_DATE_EPOCH}" \
       --owner=0 --group=0 --numeric-owner \
       --pax-option=delete=atime,delete=ctime \
       -cf /opt/site-packages.tar -C /opt/site-packages . \
    && touch -h -d "@${SOURCE_DATE_EPOCH}" /opt/site-packages.tar


FROM python:3.11-slim@sha256:00af38ae2ed311628970782e8a2d7f014d8909dbc63cb97bc0a158187f4db045

ARG SOURCE_REPOSITORY=quali-journal-track-a-clean-standalone
ARG SOURCE_BRANCH=track-a-07s-static-closure-proofpack
ARG SOURCE_COMMIT
ARG SOURCE_DATE_EPOCH

LABEL source_repository="${SOURCE_REPOSITORY}" \
      source_branch="${SOURCE_BRANCH}" \
      source_commit="${SOURCE_COMMIT}" \
      task_id="R9ZNW-469A" \
      target_service="qlib-skillup-runtime" \
      target_mode="authenticated_limited_field_beta"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/tmp/qlib-site-packages \
    PORT=8080

WORKDIR /app

COPY --from=dependencies --chown=0:0 /opt/site-packages.tar /opt/site-packages.tar

COPY --chown=10001:10001 admin/__init__.py /app/admin/__init__.py
COPY --chown=10001:10001 admin/f13_bridge_api.py /app/admin/f13_bridge_api.py
COPY --chown=10001:10001 admin/f13_library_db_evidence_retrieval.py /app/admin/f13_library_db_evidence_retrieval.py
COPY --chown=10001:10001 admin/f13_library_safe_metadata_sidecar_registry.py /app/admin/f13_library_safe_metadata_sidecar_registry.py
COPY --chown=10001:10001 admin/f13_runtime_guard.py /app/admin/f13_runtime_guard.py
COPY --chown=10001:10001 admin/f13_skillup_answer_hold_adapter.py /app/admin/f13_skillup_answer_hold_adapter.py
COPY --chown=10001:10001 admin/f13_skillup_bridge.py /app/admin/f13_skillup_bridge.py
COPY --chown=10001:10001 admin/dist/ /app/runtime/dist/
COPY --chown=10001:10001 data/library/ /app/data/library/
COPY --chown=10001:10001 deploy/qlib-skillup-runtime/runtime.py /app/runtime/runtime.py

USER 10001:10001
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import json,urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3); assert r.status == 200 and json.load(r).get('status') == 'ok'"]

CMD ["sh", "-c", "mkdir -p /tmp/qlib-site-packages && tar -xf /opt/site-packages.tar -C /tmp/qlib-site-packages && exec python -m uvicorn runtime:app --app-dir /app/runtime --host 0.0.0.0 --port 8080 --no-access-log"]
