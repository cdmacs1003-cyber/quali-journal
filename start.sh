#!/bin/sh
#
# Entry point for the QualiJournal service.  This script is invoked by
# Cloud Run and ensures that our FastAPI application is started with
# Uvicorn bound to the port provided via the PORT environment variable.
#
# We deliberately avoid any dynamic path detection here – the FastAPI
# application lives in ``server_quali.py`` at the repository root and
# exposes an ``app`` object that Uvicorn can import.  Cloud Run sets
# the PORT variable at runtime; default to 8080 for local testing.
set -eu

# Honour the PORT environment variable or fall back to 8080
PORT="${PORT:-8080}"

# Start Uvicorn via the Python module invocation.  Using ``python -m``
# ensures the correct Python interpreter is used even when installed in
# a virtual environment.  ``exec`` replaces the shell with the uvicorn
# process so signals are propagated correctly.
exec python -m uvicorn server_quali:app --host 0.0.0.0 --port "${PORT}" --workers 1
