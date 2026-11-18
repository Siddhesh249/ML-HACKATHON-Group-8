#!/usr/bin/env python3
"""Stage 5 — classifier-only vs classifier + HMM on the student-test trips."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import config  # noqa: E402
from run_eval import run_evaluation  # noqa: E402


def main() -> int:
    frame = run_evaluation(
        data_root=str(config.DATA_ROOT),
        artifact_dir=str(config.ARTIFACT_DIR),
        partition="student_test",
        modes=["hmm", "classifier"],
        out_path=str(config.RESULTS_DIR / "hmm" / "classifier_vs_hmm.csv"),
    )
    print(f"Wrote {len(frame)} rows to results/hmm/classifier_vs_hmm.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
