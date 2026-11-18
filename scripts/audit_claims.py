#!/usr/bin/env python3
"""T34 — claim/evidence audit: every reported number must map to a results file.

Checks, in order:

1. **Numbers ledger** (`viva.md` §9): for every row, the cited source file(s)
   exist, and every number in the *Value* cell can be found in those files
   (allowing for rounding to the quoted precision, percentages and thousands
   separators). Rows still marked `TBD` are listed separately.
2. **Path references** in `viva.md`, `README.md`, `report/*.md` and
   `docs/*.md`: every quoted ``results/…`` or ``artifacts/…`` path exists.

A number that cannot be matched automatically is not necessarily wrong (e.g.
"34.9 min" from a value stored in seconds); it is listed for a human check.
Outputs go to ``results/audit/``. Exit code 1 if a source file or a quoted
path is missing.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM_IN_VALUE = re.compile(r"(?<![\w.`])[-−+]?\d[\d,]*(?:\.\d+)?%?")
NUM_IN_SOURCE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
PATH_REF = re.compile(r"`((?:results|artifacts)/[^`\s]+)`")
CODE_SPAN = re.compile(r"`[^`]*`")
TEXT_SUFFIXES = {".csv", ".json", ".txt", ".md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viva", default=str(ROOT / "viva.md"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "audit"))
    return parser.parse_args()


def expand(path: str) -> list[str]:
    """Expand `a/{x,y}.csv` and globs into concrete repo-relative paths."""
    m = re.search(r"\{([^}]+)\}", path)
    if m:
        out = []
        for part in m.group(1).split(","):
            out.extend(expand(path[: m.start()] + part + path[m.end():]))
        return out
    if any(ch in path for ch in "*?["):
        return [str(Path(p).relative_to(ROOT)) for p in sorted(glob.glob(str(ROOT / path)))] or [path]
    return [path]


def ledger_rows(viva: Path) -> list[dict]:
    lines = viva.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("## ") and "ledger" in l.lower())
    rows = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if not line.startswith("|") or set(line.replace("|", "").strip()) <= {"-", " "}:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0].lower() == "claim" or len(cells) < 3:
            continue
        rows.append({"claim": cells[0], "value": cells[1], "source": cells[2],
                     "owner": cells[3] if len(cells) > 3 else "", "status": cells[4] if len(cells) > 4 else ""})
    return rows


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


def source_numbers(paths: list[str]) -> list[tuple[float, str]]:
    """(value, label) pairs: every number in the sources, labelled by file and
    column/key, plus simple derived values a claim may quote (CSV column
    mean/max, row and category counts, ms→s and s→min conversions)."""
    import pandas as pd

    out: list[tuple[float, str]] = []
    for p in paths:
        full = ROOT / p
        if full.suffix.lower() not in TEXT_SUFFIXES or not full.exists():
            continue
        name = full.name
        if full.suffix.lower() == ".csv":
            frame = pd.read_csv(full)
            out.append((float(len(frame)), f"{name}:rows"))
            for col in frame.select_dtypes("number").columns:
                vals = frame[col].dropna()
                out.extend((float(v), f"{name}:{col}") for v in vals)
                if len(vals):
                    out += [(float(vals.mean()), f"{name}:mean({col})"), (float(vals.max()), f"{name}:max({col})")]
            for col in list(frame.select_dtypes(exclude="number").columns)[:4]:
                out.extend((float(n), f"{name}:count({col}={k})") for k, n in frame[col].value_counts().items())
        elif full.suffix.lower() == ".json":
            out.extend((v, f"{name}:{k}") for k, v in _flatten(json.loads(full.read_text(encoding="utf-8"))))
        else:
            text = full.read_text(encoding="utf-8", errors="ignore")
            out.extend((float(t), f"{name}:text") for t in NUM_IN_SOURCE.findall(text))
    # Unit conversions only where the unit makes them meaningful.
    derived = [(v / 1000.0, f"{lab}/1000") for v, lab in out if "ms" in lab.split(":")[-1]]
    derived += [(v / 60.0, f"{lab}/60") for v, lab in out if "second" in lab.split(":")[-1]]
    return out + derived


def matches(token: str, nums: list[tuple[float, str]]) -> str | None:
    """Label of the first source value that rounds to ``token``; None if none."""
    tok = token.replace("−", "-").replace(",", "")
    pct = tok.endswith("%")
    tok = tok.rstrip("%")
    try:
        value = float(tok)
    except ValueError:
        return "non-numeric"
    decimals = len(tok.split(".")[1]) if "." in tok else 0
    for x, label in nums:
        if round(abs(x), decimals) == round(abs(value), decimals):
            return label
        if pct and round(abs(x) * 100, decimals) == round(abs(value), decimals):
            return f"{label}×100"
    return None


def audit_ledger(viva: Path) -> list[dict]:
    results = []
    for row in ledger_rows(viva):
        sources = [p for tok in PATH_REF.findall(row["source"]) for p in expand(tok)]
        missing = [p for p in sources if not (ROOT / p).exists()]
        is_tbd = "TBD" in row["value"] or row["status"].upper().startswith("TBD")
        tokens = NUM_IN_VALUE.findall(CODE_SPAN.sub("", row["value"]))
        nums = source_numbers([p for p in sources if p not in missing])
        found = {t: matches(t, nums) for t in tokens} if nums else {t: None for t in tokens}
        unmatched = [t for t, lab in found.items() if lab is None]
        evidence = "; ".join(f"{t} ← {lab}" for t, lab in found.items() if lab)
        if is_tbd:
            verdict = "TBD"
        elif not sources:
            verdict = "NO_SOURCE"
        elif missing:
            verdict = "MISSING_SOURCE"
        elif not tokens or not unmatched:
            verdict = "VERIFIED"
        else:
            verdict = "CHECK_MANUALLY"
        results.append({**row, "sources": ";".join(sources), "missing_sources": ";".join(missing),
                        "n_numbers": len(tokens), "unmatched_numbers": " ".join(unmatched), "evidence": evidence,
                        "verdict": verdict})
    return results


def classify(file: Path, lines: list[str], lineno: int, path: str, pending: set[str]) -> str:
    if path in pending:
        return "PENDING"          # a deliverable of a task whose ledger row is still TBD
    if file.name == "README.md" and re.match(r"\|\s*T\d+\s*\|", lines[lineno - 1]):
        return "PLANNED"          # task-board row naming a future deliverable
    if file.name == "AGENT_LOG.md":
        return "HISTORICAL"       # append-only log; never rewritten
    if file.name == "STATUS.md" and any(l.startswith("### Handoff") for l in lines[:lineno]):
        return "HISTORICAL"       # dated handoff snapshot
    return "BROKEN"


def audit_paths(pending: set[str]) -> list[dict]:
    files = [ROOT / "viva.md", ROOT / "README.md", *sorted((ROOT / "report").glob("*.md")),
             *sorted((ROOT / "docs").glob("*.md"))]
    out = []
    for f in files:
        if not f.exists():
            continue
        lines = f.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, 1):
            for ref in PATH_REF.findall(line):
                ref = ref.rstrip(".,;:)")
                if "<" in ref or ref.endswith("/"):
                    continue  # templates such as results/<date>_… or directory mentions
                for p in expand(ref):
                    if not (ROOT / p).exists():
                        out.append({"file": str(f.relative_to(ROOT)), "line": lineno, "path": p,
                                    "kind": classify(f, lines, lineno, p, pending)})
    return out


def stale_tbd_tables() -> list[dict]:
    """`| TBD |` cells outside the viva ledger (duplicate claim tables)."""
    out = []
    for f in [ROOT / "README.md", *sorted((ROOT / "report").glob("*.md"))]:
        if not f.exists():
            continue
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if line.startswith("|") and re.search(r"\|\s*TBD\s*\|", line):
                out.append({"file": str(f.relative_to(ROOT)), "line": lineno, "row": line[:120]})
    return out


def main() -> int:
    args = parse_args()
    import pandas as pd

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = pd.DataFrame(audit_ledger(Path(args.viva)))
    ledger.to_csv(out / "claims_audit.csv", index=False)
    pending = {p for _, r in ledger[ledger["verdict"] == "TBD"].iterrows() for p in r["sources"].split(";") if p}
    refs = pd.DataFrame(audit_paths(pending), columns=["file", "line", "path", "kind"])
    refs.to_csv(out / "path_refs_not_found.csv", index=False)
    broken = refs[refs["kind"] == "BROKEN"]
    tbd_tables = pd.DataFrame(stale_tbd_tables(), columns=["file", "line", "row"])

    counts = ledger["verdict"].value_counts().to_dict()
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(["python3", "scripts/audit_claims.py", *sys.argv[1:]]),
        "ledger_rows": int(len(ledger)),
        "verdicts": counts,
        "tbd_rows": ledger.loc[ledger["verdict"] == "TBD", ["claim", "owner"]].to_dict("records"),
        "check_manually": ledger.loc[ledger["verdict"] == "CHECK_MANUALLY",
                                     ["claim", "owner", "unmatched_numbers"]].to_dict("records"),
        "missing_sources": ledger.loc[ledger["verdict"] == "MISSING_SOURCE",
                                      ["claim", "missing_sources"]].to_dict("records"),
        "path_refs_not_found": refs["kind"].value_counts().to_dict(),
        "broken_path_refs": broken[["file", "line", "path"]].to_dict("records"),
        "tbd_outside_ledger": tbd_tables.to_dict("records"),
        "clean": bool(counts.get("TBD", 0) == 0 and counts.get("MISSING_SOURCE", 0) == 0
                      and counts.get("NO_SOURCE", 0) == 0 and len(broken) == 0 and tbd_tables.empty),
    }
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    hard_fail = counts.get("MISSING_SOURCE", 0) + counts.get("NO_SOURCE", 0) + len(broken)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
