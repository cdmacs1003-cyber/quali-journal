from qj_paths import rel as qj_rel
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_quali_today.py ??Cross?몆latform daily one?멵lick pipeline for QualiJournal.

This script replicates the functionality of the original PowerShell script
(`run_quali_today.ps1`) but runs entirely in Python, making it portable
across operating systems.  It performs the following sequence of steps:

1. Collect official sources (smt/supply) via orchestrator.
2. Collect community candidates.
3. Collect keyword?몊pecific articles with optional external RSS.
4. Rebuild the working set from today's selections.
5. Augment from the official pool to guarantee at least 20 items and 15 approvals.
6. Normalize selection files (wrap lists into dicts).
7. Sync approved items into ``selected_articles.json``.
8. Check the number of approvals and, if below the configured gate,
   temporarily disable the gate, publish, and then restore the gate.

Usage:
    python run_quali_today.py --keyword "IPC-A-610" [--use-external-rss]

The keyword defaults to ``IPC-A-610``.  When ``--use-external-rss`` is
supplied, the keyword collection step includes external RSS feeds.

Note: The script must be executed from the project root so that relative
paths resolve correctly.  It will change the working directory to the
directory containing this file to avoid path issues.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


def run_subprocess(args: list[str], cwd: Path) -> int:
    """Run a subprocess synchronously and return its exit code.

    The subprocess is executed using the current Python interpreter and
    inherits UTF?? encoding for all standard streams to avoid decoding
    errors on Windows consoles.  If the subprocess fails, a message is
    printed to stderr.

    Args:
        args: Command arguments (including the script name and its options).
        cwd: Working directory in which to execute the command.

    Returns:
        int: The subprocess's exit code.
    """
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        result = subprocess.run(
            [sys.executable] + args,
            cwd=str(cwd),
            encoding="utf-8",
            errors="replace",
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            # Print stderr to aid troubleshooting but do not stop execution
            print(result.stderr.strip(), file=sys.stderr)
        return result.returncode
    except Exception as exc:
        print(f"[ERROR] Failed to run {' '.join(args)}: {exc}", file=sys.stderr)
        return 1


def update_gate(cfg_path: Path, enabled: bool) -> None:
    """Enable or disable the editor approval gate in config.json.

    This helper modifies the ``features.require_editor_approval`` key of
    ``config.json``.  If the file does not exist or cannot be parsed,
    nothing is changed.

    Args:
        cfg_path: Path to the configuration file.
        enabled: If True, enable the gate; otherwise disable it.
    """
    try:
        if not cfg_path.exists():
            return
        with cfg_path.open("r", encoding="utf-8") as rf:
            cfg = json.load(rf)
        # Ensure the features section exists
        if not isinstance(cfg, dict):
            cfg = {}
        features = cfg.get("features") or {}
        features["require_editor_approval"] = bool(enabled)
        cfg["features"] = features
        # Write back the config atomically
        tmp = cfg_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(cfg_path)
    except Exception as exc:
        print(f"[WARN] Failed to {'enable' if enabled else 'disable'} gate: {exc}")


def count_approved(path: Path) -> int:
    """Count approved articles in ``data/selected_keyword_articles.json``.

    Args:
        path: Path to the keyword selection JSON file.

    Returns:
        int: Number of articles with ``approved`` set to True.
    """
    try:
        with path.open("r", encoding="utf-8") as rf:
            j = json.load(rf)
        arts = j.get("articles") if isinstance(j, dict) else j
        if not isinstance(arts, list):
            return 0
        return sum(1 for a in arts if isinstance(a, dict) and a.get("approved"))
    except Exception:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keyword",
        "-k",
        default="IPC-A-610",
        help="Keyword for the daily run (default: IPC-A-610)",
    )
    parser.add_argument(
        "--use-external-rss",
        action="store_true",
        help="Include external RSS feeds when collecting keyword articles",
    )
    args = parser.parse_args()

    # Determine project root as the directory containing this script
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir
    # Ensure we execute relative commands from the project root
    os.chdir(str(root_dir))

    print("==[1/7] Collecting official and community sources==")
    # Official collection
    run_subprocess(["orchestrator.py", "--collect"], cwd=root_dir)
    # Community collection
    run_subprocess(["orchestrator.py", "--collect-community"], cwd=root_dir)
    # Keyword collection
    kw_args = ["orchestrator.py", "--collect-keyword", args.keyword]
    if args.use_external_rss:
        kw_args.append("--use-external-rss")
    run_subprocess(kw_args, cwd=root_dir)

    print("==[2/7] Rebuilding working set from today's selections==")
    run_subprocess(["tools/rebuild_kw_from_today.py", args.keyword], cwd=root_dir)

    print("==[3/7] Augmenting from official pool to fill to 20 and ensure 15 approvals==")
    run_subprocess(["tools/augment_from_official_pool.py"], cwd=root_dir)

    print("==[4/7] Normalizing publish sources (wrap lists into dict+articles)==")
    run_subprocess(["tools/repair_selection_files.py"], cwd=root_dir)

    print("==[5/7] Syncing approved articles to selected_articles.json==")
    run_subprocess(["tools/sync_selected_for_publish.py"], cwd=root_dir)

    # Count approvals
    approved_count = count_approved(root_dir / "data" / "selected_keyword_articles.json")
    print(f"Current approved count: {approved_count}")

    cfg_path = root_dir / "config.json"
    # Determine gate threshold from config
    gate_required = 15
    require_editor = True
    try:
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            gate_required = int(
                cfg.get("gate_required")
                or cfg.get("features", {}).get("gate_required")
                or gate_required
            )
            require_editor = bool(cfg.get("features", {}).get("require_editor_approval", True))
    except Exception:
        pass

    emergency = approved_count < gate_required and require_editor

    if emergency:
        print(
            "==[6/7] Emergency: approval below gate ??temporarily disabling approval requirement=="
        )
        update_gate(cfg_path, enabled=False)
        print("Gate disabled; publishing keyword page in emergency mode")
        run_subprocess(["orchestrator.py", "--publish-keyword", args.keyword], cwd=root_dir)
        # Restore gate
        update_gate(cfg_path, enabled=True)
        print("Gate restored to enabled state")
    else:
        print("==[6/6] Publishing keyword page (normal mode)==")
        run_subprocess(["orchestrator.py", "--publish-keyword", args.keyword], cwd=root_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
