import numpy as np

from pathsnap import candidates, hmm, io, labels
from pathsnap.contracts import CandidateSet, Network, Track, TripData, TripMeta


def _candidate_set(arc_ids, ratios, offsets=None):
    arc_ids = np.asarray(arc_ids, dtype=np.int64)
    ratios = np.asarray(ratios, dtype=float)
    if offsets is None:
        offsets = np.arange(arc_ids.size + 1, dtype=np.int64)
    return CandidateSet(
        offsets=np.asarray(offsets, dtype=np.int64),
        arc_ids=arc_ids,
        distance_m=np.ones(arc_ids.size, dtype=float),
        projection_ratio=ratios,
        projection_xy=np.zeros((arc_ids.size, 2), dtype=float),
        arc_heading_deg=np.zeros(arc_ids.size, dtype=float),
        origin=(8.0, 49.0),
    )


def _trip(nodes, arcs, times=(1.0, 2.0, 3.0)):
    nodes = np.asarray(nodes, dtype=float)
    arcs = np.asarray(arcs, dtype=np.int64)
    track = Track(
        lonlat=np.asarray([[8.0001, 49.0], [8.0009, 49.0], [8.0018, 49.0]], dtype=float),
        t=np.asarray(times, dtype=float),
    )
    meta = TripMeta(
        trip_id="synthetic-hmm",
        partition="student_test",
        directory=None,
        n_points=len(track),
        n_nodes=nodes.shape[0],
        n_arcs=arcs.shape[0],
        n_route_arcs=0,
        has_route=False,
        bbox=(8.0, 8.002, 49.0, 49.001),
    )
    return TripData(meta=meta, track=track, network=Network(nodes=nodes, arcs=arcs))


def test_viterbi_prefers_consistent_path():
    emissions = [np.array([0.0, -10.0]), np.array([-20.0, 0.0])]
    transitions = [np.array([[0.0, -10.0], [-10.0, 0.0]])]
    states, score = hmm.viterbi_layers(emissions, transitions)
    assert states == [0, 1]
    assert np.isfinite(score)


def test_viterbi_transition_can_override_emission():
    # Emission prefers staying on state 0, but the transition matrix forces a swap.
    emissions = [np.array([0.0, -5.0]), np.array([0.0, -0.1])]
    transitions = [np.array([[-10.0, 0.0], [0.0, -10.0]])]
    states, _ = hmm.viterbi_layers(emissions, transitions)
    assert states == [0, 1]


def test_classifier_choice_and_hmm_on_synthetic_trip(synthetic_root):
    root, trip_id, route = synthetic_root
    trip = io.load_trip(root / "student_test" / trip_id, with_route=True)
    generator = candidates.CandidateGenerator(trip.network, radius_m=40.0, top_k=8)
    cand = generator.generate(trip.track)
    probs = np.full(len(cand.arc_ids), 0.05)
    mask = np.isin(cand.arc_ids, route)
    probs[mask] = 0.9

    arcs, chosen = hmm.classifier_choice(cand, probs)
    assert arcs.shape == (len(trip.track),)
    assert chosen.shape == arcs.shape

    cfg = hmm.HMMConfig(lam=0.2, sigma_m=10.0, beta_m=20.0)
    decoded, _ = hmm.decode_hmm(trip, cand, probs, cfg)
    matched = decoded[decoded >= 0]
    assert matched.size > 0
    assert np.mean(np.isin(matched, route)) > 0.9
    # The synthetic network allows a connected route along the middle row.
    continuity = np.mean([int(x) in route for x in matched])
    assert continuity > 0.9


def test_emission_normalisation_is_finite():
    cand = CandidateSet(
        offsets=np.array([0, 3]),
        arc_ids=np.array([1, 2, 3]),
        distance_m=np.array([1.0, 100.0, 2.0]),
        projection_ratio=np.zeros(3),
        projection_xy=np.zeros((3, 2)),
        arc_heading_deg=np.zeros(3),
    )
    scores = hmm.emission_scores(cand, np.array([0.0, 0.0, 0.0]), 0, hmm.HMMConfig())
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_transition_prefers_forward_same_arc_progress(synthetic_root):
    root, trip_id, _ = synthetic_root
    trip = io.load_trip(root / "student_test" / trip_id, with_route=True)
    arc = int(np.flatnonzero((trip.network.arcs[:, 0] == 3) & (trip.network.arcs[:, 1] == 4))[0])
    next_arc = int(np.flatnonzero((trip.network.arcs[:, 0] == 4) & (trip.network.arcs[:, 1] == 5))[0])
    cand = _candidate_set([arc, arc, next_arc], [0.25, 0.75, 0.0], offsets=[0, 1, 3])
    matrix = hmm.DecoderContext(trip, cand, hmm.HMMConfig()).transition(0, 1)
    assert matrix.shape == (1, 2)
    assert matrix[0, 0] > matrix[0, 1]


def test_transition_reversed_progress_is_not_preferred(synthetic_root):
    root, trip_id, _ = synthetic_root
    trip = io.load_trip(root / "student_test" / trip_id, with_route=True)
    arc = int(np.flatnonzero((trip.network.arcs[:, 0] == 3) & (trip.network.arcs[:, 1] == 4))[0])
    next_arc = int(np.flatnonzero((trip.network.arcs[:, 0] == 4) & (trip.network.arcs[:, 1] == 5))[0])
    cand = _candidate_set([arc, arc, next_arc], [0.75, 0.25, 0.0], offsets=[0, 1, 3])
    matrix = hmm.DecoderContext(trip, cand, hmm.HMMConfig()).transition(0, 1)
    assert matrix[0, 0] <= -1e8
    assert np.isfinite(matrix[0, 1])


def test_transition_marks_unreachable_candidate_infeasible():
    trip = _trip(
        [[8.0, 49.0], [8.001, 49.0], [8.002, 49.0], [8.0, 49.0005], [8.001, 49.0005]],
        [[0, 1], [1, 2], [3, 4]],
    )
    cand = _candidate_set([0, 1, 2], [0.5, 0.0, 0.5], offsets=[0, 1, 3])
    matrix = hmm.DecoderContext(trip, cand, hmm.HMMConfig()).transition(0, 1)
    assert matrix[0, 1] <= -1e8


def test_transition_gap_reset_has_equal_scores():
    trip = _trip([[8.0, 49.0], [8.001, 49.0]], [[0, 1]], times=(1.0, 100.0, 101.0))
    cand = _candidate_set([0, 0], [0.1, 0.9], offsets=[0, 1, 2])
    matrix = hmm.DecoderContext(trip, cand, hmm.HMMConfig()).transition(0, 1)
    assert np.allclose(matrix, -np.log(1.0))


def test_uturn_penalty_reduces_reverse_transition():
    trip = _trip(
        [[8.0, 49.0], [8.001, 49.0], [8.002, 49.0]],
        [[0, 1], [1, 0], [1, 2]],
    )
    cand = _candidate_set([0, 1, 2], [0.5, 0.5, 0.0], offsets=[0, 1, 3])
    no_penalty = hmm.DecoderContext(trip, cand, hmm.HMMConfig(uturn_penalty=0.0)).transition(0, 1)
    penalty = hmm.DecoderContext(trip, cand, hmm.HMMConfig(uturn_penalty=5.0)).transition(0, 1)
    assert penalty[0, 0] < no_penalty[0, 0]


def test_empty_candidate_layer_is_skipped_by_decoder(synthetic_root):
    root, trip_id, _ = synthetic_root
    trip = io.load_trip(root / "student_test" / trip_id, with_route=True)
    arc = int(np.flatnonzero((trip.network.arcs[:, 0] == 3) & (trip.network.arcs[:, 1] == 4))[0])
    cand = _candidate_set([arc, arc], [0.2, 0.8], offsets=[0, 1, 1, 2])
    decoded, _ = hmm.decode_hmm(trip, cand, np.array([0.9, 0.8]))
    assert decoded[1] == -1
    assert decoded[[0, 2]].tolist() == [arc, arc]
