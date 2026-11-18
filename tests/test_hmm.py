import numpy as np

from pathsnap import candidates, hmm, io, labels
from pathsnap.contracts import CandidateSet


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
