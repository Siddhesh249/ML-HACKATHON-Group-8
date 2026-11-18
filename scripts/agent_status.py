#!/usr/bin/env python3
"""Mechanical status snapshot for humans and agents.

Usage:
    python scripts/agent_status.py
    python scripts/agent_status.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIVA = ROOT / "viva.md"
README = ROOT / "README.md"
LOG = ROOT / "docs" / "AGENT_LOG.md"
STATUS = ROOT / "docs" / "STATUS.md"
DATA = ROOT / "data"
CACHE = ROOT / "cache"
ARTIFACTS = ROOT / "artifacts"
REQUIRED = (".track", ".nodes", ".arcs")


def sh(*args: str) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def git_info() -> dict:
    branch = sh("git", "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    commit = sh("git", "rev-parse", "--short", "HEAD") or "unknown"
    subject = sh("git", "log", "-1", "--pretty=%s")
    dirty = sh("git", "status", "--porcelain")
    dirty_files = [line for line in dirty.splitlines() if line.strip()]
    return {
        "branch": branch,
        "commit": commit,
        "last_commit": subject,
        "dirty_files": len(dirty_files),
        "dirty_sample": dirty_files[:10],
    }


def dataset_info() -> dict | None:
    path = CACHE / "dataset.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    return {
        "n_trips": meta.get("n_trips"),
        "n_observations": meta.get("n_observations"),
        "n_rows": meta.get("n_rows"),
        "n_features": meta.get("n_features"),
        "path": str(path),
    }


def artifact_info() -> dict | None:
    path = ARTIFACTS / "model_card.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        card = json.load(fh)
    return {
        "model": card.get("model_name"),
        "version": card.get("model_version"),
        "git_commit": card.get("git_commit"),
        "path": str(path),
    }


def partition_counts() -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for part in ("train", "student_test", "evaluator_test"):
        folder = DATA / part
        if not folder.is_dir():
            counts[part] = None
            continue
        try:
            counts[part] = sum(
                1
                for d in folder.iterdir()
                if d.is_dir() and all((d / f"{d.name}{s}").exists() for s in REQUIRED)
            )
        except OSError:
            counts[part] = None
    return counts


def counts_in(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return len(re.findall(pattern, path.read_text(encoding="utf-8", errors="ignore")))


def code_todos() -> int:
    total = 0
    for folder in ("pathsnap", "backend", "ui", "scripts", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            try:
                total += path.read_text(encoding="utf-8", errors="ignore").count("TODO")
            except OSError:
                pass
    return total


def last_log_entries(n: int = 3) -> list[str]:
    if not LOG.exists():
        return []
    headings = [line.strip("# ").strip() for line in LOG.read_text(encoding="utf-8").splitlines() if line.startswith("## AL-")]
    return headings[-n:]


def status_headings() -> list[str]:
    if not STATUS.exists():
        return []
    return [line.strip("# ").strip() for line in STATUS.read_text(encoding="utf-8").splitlines() if line.startswith("## ")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    snapshot = {
        "git": git_info(),
        "dataset": dataset_info(),
        "artifact": artifact_info(),
        "partitions": partition_counts(),
        "task_board": {
            "done": counts_in(README, "✅"),
            "in_progress": counts_in(README, "🟨"),
            "todo": counts_in(README, "⬜"),
        },
        "viva": {
            "tbd": counts_in(VIVA, r"\bTBD\b"),
            "has_ledger": "Numbers ledger" in VIVA.read_text(encoding="utf-8") if VIVA.exists() else False,
        },
        "code_todos": code_todos(),
        "last_log_entries": last_log_entries(),
        "status_sections": status_headings(),
    }

    if args.json:
        print(json.dumps(snapshot, indent=2))
        return 0

    g = snapshot["git"]
    print("=" * 72)
    print("PathSnap — agent status")
    print("=" * 72)
    print(f"git       : {g['branch']} @ {g['commit']} — {g['last_commit']}")
    print(f"dirty     : {g['dirty_files']} file(s)")
    for line in g["dirty_sample"]:
        print(f"            {line}")
    print(f"partitions: {snapshot['partitions']}")
    ds = snapshot["dataset"]
    print(f"dataset   : {ds if ds else 'missing (run make dataset)'}")
    art = snapshot["artifact"]
    print(f"artifact  : {art if art else 'missing (run make export)'}")
    print(
        "tasks     : "
        f"done={snapshot['task_board']['done']} "
        f"in_progress={snapshot['task_board']['in_progress']} "
        f"todo={snapshot['task_board']['todo']}"
    )
    print(f"viva TBDs : {snapshot['viva']['tbd']} (numbers ledger: {snapshot['viva']['has_ledger']})")
    print(f"code TODOs: {snapshot['code_todos']}")
    print("\nLast agent-log entries (newest last):")
    for entry in snapshot["last_log_entries"]:
        print(f"  - {entry}")
    print("\nRead next:")
    print("  docs/STATUS.md  →  docs/AGENT_LOG.md (last 3)  →  viva.md")
    print("  then claim your task in docs/STATUS.md before editing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
