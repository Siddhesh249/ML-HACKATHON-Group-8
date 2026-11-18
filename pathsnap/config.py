"""Central configuration for PathSnap.

Everything tunable that is shared between training and inference lives here so
that the exported artifact and the API cannot drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT", ROOT / "data"))
ARTIFACT_DIR = Path(os.environ.get("PATHSNAP_ARTIFACT_DIR", ROOT / "artifacts"))
CACHE_DIR = Path(os.environ.get("PATHSNAP_CACHE_DIR", ROOT / "cache"))
RESULTS_DIR = Path(os.environ.get("PATHSNAP_RESULTS_DIR", ROOT / "results"))

EVALUATION_PARTITION = os.environ.get("PATHSNAP_EVALUATION_PARTITION", "evaluator_test")
FALLBACK_PARTITION = os.environ.get("PATHSNAP_FALLBACK_PARTITION", "student_test")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = int(os.environ.get("PATHSNAP_SEED", "42"))

# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
CANDIDATE_RADIUS_M = 60.0
CANDIDATE_RADIUS_FALLBACKS_M = (100.0, 150.0)
CANDIDATE_TOP_K = 15
ARC_SAMPLE_STEP_M = 5.0
ARC_MAX_SAMPLES = 200

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABEL_MAX_DISTANCE_M = 30.0

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
SMOOTH_WINDOW = 5  # odd number of observations

# ---------------------------------------------------------------------------
# HMM defaults (tuned values are stored in artifacts/hmm_config.json)
# ---------------------------------------------------------------------------
HMM_GAP_RESET_S = 30.0
HMM_MAX_SPEED_MPS = 55.0  # ~200 km/h, generous upper bound
HMM_BETA_M = 10.0
HMM_SIGMA_M = 8.0
HMM_LAMBDA = 0.5
HMM_UTURN_PENALTY = 2.0

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------
# Order is part of the artifact contract. Adding a feature requires bumping
# FEATURE_VERSION and re-exporting the model.
FEATURE_VERSION = "v1"
FEATURE_NAMES: tuple[str, ...] = (
    "log_distance",
    "distance_ratio_best",
    "candidate_rank",
    "projection_ratio",
    "abs_heading_diff",
    "cos_heading_diff",
    "log_arc_length",
    "log_dist_to_arc_start",
    "log_dist_to_arc_end",
    "node_degree_from",
    "node_degree_to",
    "has_reciprocal_arc",
    "log_graph_dist_prev",
    "same_arc_as_prev",
    "log_graph_minus_gps_displacement",
    "log_speed",
    "log_gap_seconds",
    "candidate_count",
)

USE_GRAPH_FEATURES = True


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
