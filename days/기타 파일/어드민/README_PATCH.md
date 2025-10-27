# QualiJournal Mid‑term Patch — 2025-10-06

## Patched files
- server_quali.py  → adds `/api/tasks/*` async endpoints, dynamic `gate_required` + `require_editor_approval` in `/api/status`, and a PATCH endpoint `/api/config/gate_required`.
- index.html       → already contains theme toggle + fallback to `/api/flow/*` when `/api/tasks/*` is absent.
- config.json      → ensure it contains `"gate_required"` and `"features.require_editor_approval"`.

## How to apply (safe & simple)
1. **Backup** the originals:
   ```
   copy server_quali.py server_quali.backup.py
   copy config.json config.backup.json
   ```
2. **Replace** `server_quali.py` with the patched file from this zip.
3. **Check** `config.json` has:
   ```json
   {
     "gate_required": 15,
     "features": {"require_editor_approval": True}
   }
   ```
4. **Run** the admin server:
   ```
   uvicorn server_quali:app --host 0.0.0.0 --port 8010 --reload
   ```
5. Open **index.html** in your browser or http://127.0.0.1:8010/

## Verifying
- `GET /api/status` now shows dynamic `gate_required` from `config.json`.
- `GET /api/tasks/recent` returns an empty list (no errors) → UI will auto‑detect async support.
- `POST /api/tasks/flow/daily` returns `{"job_id": "..."},` and `/api/tasks/{job_id}` shows live steps after a moment.

## Rollback
Restore `server_quali.py` from `server_quali.backup.py`.