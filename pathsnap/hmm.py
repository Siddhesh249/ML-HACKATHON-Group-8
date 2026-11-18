"""HMM emissions, network-aware transitions and Viterbi decoding."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import config, geo, network as net
from .contracts import CandidateSet, Prediction, TripData


@dataclass
class HMMConfig:
    beta_m: float = config.HMM_BETA_M
    sigma_m: float = config.HMM_SIGMA_M
    lam: float = config.HMM_LAMBDA
    max_speed_mps: float = config.HMM_MAX_SPEED_MPS
    gap_reset_s: float = config.HMM_GAP_RESET_S
    uturn_penalty: float = config.HMM_UTURN_PENALTY
    transition_slack_m: float = 300.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Generic Viterbi
# ---------------------------------------------------------------------------
def viterbi_layers(
    log_emissions: list[np.ndarray],
    log_transitions: list[np.ndarray],
) -> tuple[list[int], float]:
    """Viterbi for layers with possibly different state counts.

    ``log_transitions[t]`` has shape ``(K_t, K_{t+1})``.
    """
    if not log_emissions:
        return [], float("-inf")
    n_layers = len(log_emissions)
    delta = np.asarray(log_emissions[0], dtype=float).copy()
    backpointers: list[np.ndarray] = []

    for t in range(1, n_layers):
        trans = np.asarray(log_transitions[t - 1], dtype=float)
        scores = delta[:, None] + trans
        best_prev = np.argmax(scores, axis=0)
        delta = scores[best_prev, np.arange(scores.shape[1])] + np.asarray(log_emissions[t], dtype=float)
        backpointers.append(best_prev)

    states = [int(np.argmax(delta))]
    for t in range(n_layers - 2, -1, -1):
        states.append(int(backpointers[t][states[-1]]))
    states.reverse()
    return states, float(np.max(delta))


# ---------------------------------------------------------------------------
# Emissions and transitions
# ---------------------------------------------------------------------------
def emission_scores(
    candidates: CandidateSet,
    probs: np.ndarray,
    obs_index: int,
    cfg: HMMConfig,
) -> np.ndarray:
    lo, hi = int(candidates.offsets[obs_index]), int(candidates.offsets[obs_index + 1])
    p = np.clip(np.asarray(probs[lo:hi], dtype=float), 1e-9, None)
    total = float(p.sum())
    p = p / total if total > 0 else np.full(p.size, 1.0 / max(p.size, 1))
    dist = np.asarray(candidates.distance_m[lo:hi], dtype=float)
    return np.log(p) - cfg.lam * (dist**2) / (2.0 * cfg.sigma_m**2)


class DecoderContext:
    """Cached arc geometry and corridor graph for one trip."""

    def __init__(
        self,
        trip: TripData,
        candidates: CandidateSet,
        cfg: HMMConfig,
        corridor_margin_m: float = 600.0,
    ) -> None:
        self.trip = trip
        self.candidates = candidates
        self.cfg = cfg
        self.arcs = trip.network.arcs
        self.a_xy, self.b_xy, self.lengths, _, _ = net.arc_geometry(
            trip.network, origin=candidates.origin
        )
        self.smooth_lonlat, _, _ = geo.smooth_track(
            trip.track.lonlat, trip.track.t, window=config.SMOOTH_WINDOW
        )
        self.track_xy, _ = geo.project_lonlat(self.smooth_lonlat, origin=candidates.origin)
        corridor = net.corridor_subgraph(
            trip.network,
            self.track_xy,
            origin=candidates.origin,
            margin_m=corridor_margin_m,
        )
        self.graph = corridor["graph"]
        self.local_index = corridor["local_index"]
        self._dist_cache: dict[int, np.ndarray] = {}

    def distances_from(self, source_global: int) -> np.ndarray | None:
        source_local = int(self.local_index[source_global]) if source_global >= 0 else -1
        if source_local < 0:
            return None
        if source_local not in self._dist_cache:
            self._dist_cache[source_local] = net.shortest_distances(
                self.graph, source_local, limit_m=5000.0
            )
        return self._dist_cache[source_local]

    def transition(self, prev_obs: int, cur_obs: int) -> np.ndarray:
        pc = self.candidates
        p_lo, p_hi = int(pc.offsets[prev_obs]), int(pc.offsets[prev_obs + 1])
        c_lo, c_hi = int(pc.offsets[cur_obs]), int(pc.offsets[cur_obs + 1])
        prev_arcs = pc.arc_ids[p_lo:p_hi].astype(np.int64)
        cur_arcs = pc.arc_ids[c_lo:c_hi].astype(np.int64)
        prev_ratio = np.clip(pc.projection_ratio[p_lo:p_hi], 0.0, 1.0)
        cur_ratio = np.clip(pc.projection_ratio[c_lo:c_hi], 0.0, 1.0)
        kp, kc = prev_arcs.size, cur_arcs.size
        log_a = np.full((kp, kc), -1e9, dtype=float)

        dt = max(float(self.trip.track.t[cur_obs] - self.trip.track.t[prev_obs]), 0.0)
        if dt >= self.cfg.gap_reset_s:
            # A gap resets the temporal constraint. Return a proper log-uniform
            # transition prior rather than unnormalised equal scores.
            return np.full((kp, kc), -np.log(max(kc, 1)), dtype=float)

        gps_disp = float(np.linalg.norm(self.track_xy[cur_obs] - self.track_xy[prev_obs]))
        max_d = max(self.cfg.max_speed_mps * dt + self.cfg.transition_slack_m, 300.0)

        for i, arc_i in enumerate(prev_arcs):
            exit_offset = (1.0 - prev_ratio[i]) * self.lengths[arc_i]
            source = int(self.arcs[arc_i, 1])
            same = cur_arcs == arc_i
            d = np.full(kc, np.inf)

            # Staying on the same arc: progress along the arc, never backwards.
            if np.any(same):
                forward = (cur_ratio[same] - prev_ratio[i]) * self.lengths[arc_i]
                d[same] = np.where(forward >= 0.0, forward, np.inf)

            # Moving to a different arc: exit + shortest path + entry.
            if np.any(~same):
                dist_row = self.distances_from(source)
                if dist_row is not None:
                    targets = self.local_index[self.arcs[cur_arcs[~same], 0]]
                    valid = targets >= 0
                    diff = np.full(int(np.sum(~same)), np.inf)
                    diff[valid] = (
                        dist_row[targets[valid]]
                        + exit_offset
                        + cur_ratio[~same][valid] * self.lengths[cur_arcs[~same][valid]]
                    )
                    d[~same] = diff

            feasible = d <= max_d
            if not np.any(feasible):
                continue
            cost = -np.abs(d - gps_disp) / self.cfg.beta_m
            reverse = (self.arcs[cur_arcs, 0] == self.arcs[arc_i, 1]) & (
                self.arcs[cur_arcs, 1] == self.arcs[arc_i, 0]
            )
            cost = cost - self.cfg.uturn_penalty * reverse
            log_a[i, feasible] = cost[feasible]

        dead = np.all(log_a <= -1e8, axis=1)
        if np.any(dead):
            # An all-infeasible row would kill that state; fall back to a
            # weak uniform prior instead of dropping a potentially correct arc.
            log_a[dead] = 0.0

        row_max = log_a.max(axis=1, keepdims=True)
        log_a = log_a - (row_max + np.log(np.exp(log_a - row_max).sum(axis=1, keepdims=True)))
        return log_a


# ---------------------------------------------------------------------------
# Public decoding entry points
# ---------------------------------------------------------------------------
def classifier_choice(
    candidates: CandidateSet,
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(candidates)
    arcs = np.full(n, -1, dtype=np.int64)
    chosen_p = np.zeros(n, dtype=float)
    for i in range(n):
        lo, hi = int(candidates.offsets[i]), int(candidates.offsets[i + 1])
        if hi <= lo:
            continue
        j = lo + int(np.argmax(probs[lo:hi]))
        arcs[i] = int(candidates.arc_ids[j])
        chosen_p[i] = float(probs[j])
    return arcs, chosen_p


def decode_hmm(
    trip: TripData,
    candidates: CandidateSet,
    probs: np.ndarray,
    cfg: HMMConfig | None = None,
    context: DecoderContext | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one arc per observation plus the model probability of that arc."""
    cfg = cfg or HMMConfig()
    context = context or DecoderContext(trip, candidates, cfg)
    n = len(candidates)
    valid = [
        i
        for i in range(n)
        if int(candidates.offsets[i + 1]) > int(candidates.offsets[i])
    ]
    arcs = np.full(n, -1, dtype=np.int64)
    chosen_p = np.zeros(n, dtype=float)
    if not valid:
        return arcs, chosen_p

    emissions = [emission_scores(candidates, probs, i, cfg) for i in valid]
    transitions = [context.transition(valid[t - 1], valid[t]) for t in range(1, len(valid))]
    states, _ = viterbi_layers(emissions, transitions)

    for layer, state in enumerate(states):
        obs = valid[layer]
        lo = int(candidates.offsets[obs])
        arcs[obs] = int(candidates.arc_ids[lo + state])
        chosen_p[obs] = float(probs[lo + state])
    return arcs, chosen_p


def decode(
    trip: TripData,
    candidates: CandidateSet,
    probs: np.ndarray,
    mode: str = "hmm",
    cfg: HMMConfig | None = None,
) -> Prediction:
    if mode == "classifier":
        arcs, chosen = classifier_choice(candidates, probs)
    elif mode == "hmm":
        arcs, chosen = decode_hmm(trip, candidates, probs, cfg)
    else:
        raise ValueError(f"Unknown decode mode: {mode}")
    return Prediction(trip_id=trip.meta.trip_id, arc_ids=arcs, probabilities=chosen, mode=mode)
