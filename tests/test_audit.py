"""Tests for the T34 claim/evidence audit helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_claims", ROOT / "scripts" / "audit_claims.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_matches_rounding_percent_and_separators():
    nums = [(0.88177, "f.csv:top1"), (0.00092, "f.csv:miss"), (208417.0, "f.json:n")]
    assert audit.matches("0.882", nums) == "f.csv:top1"
    assert audit.matches("0.092%", nums) == "f.csv:miss×100"
    assert audit.matches("208,417", nums) == "f.json:n"
    assert audit.matches("0.95", nums) is None


def test_expand_braces():
    assert audit.expand("results/x/{a,b}.csv") == ["results/x/a.csv", "results/x/b.csv"]
