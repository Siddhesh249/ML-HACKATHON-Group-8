import numpy as np

from pathsnap import candidates, io, labels


def test_candidate_generation_recalls_truth(synthetic_root):
    root, trip_id, route = synthetic_root
    trip = io.load_trip(root / "student_test" / trip_id, with_route=True)
    generator = candidates.CandidateGenerator(trip.network, radius_m=40.0, top_k=8)
    candidate_set = generator.generate(trip.track)
    assert len(candidate_set) == len(trip.track)
    assert candidate_set.offsets[-1] == len(candidate_set.arc_ids)

    alignment = labels.align_track_to_route(trip)
    miss, n_eval = candidates.candidate_miss_rate(
        candidate_set, alignment.arc_ids, alignment.confidence
    )
    assert n_eval > 0
    assert miss < 0.1

    # At least one candidate per observation on this clean synthetic trace.
    counts = np.diff(candidate_set.offsets)
    assert (counts > 0).all()
    assert set(np.unique(candidate_set.arc_ids)).issuperset(set(route.tolist()))
