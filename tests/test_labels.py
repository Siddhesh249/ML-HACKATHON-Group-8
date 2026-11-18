import numpy as np

from pathsnap import io, labels


def test_alignment_recovers_reference_route(synthetic_root):
    root, trip_id, route = synthetic_root
    trip = io.load_trip(root / "student_test" / trip_id, with_route=True)
    alignment = labels.align_track_to_route(trip)

    assert alignment.coverage > 0.8
    labeled = alignment.arc_ids[alignment.confidence]
    assert labeled.size > 0
    assert set(np.unique(labeled)).issubset(set(route.tolist()))
    assert np.median(alignment.distance_m) < 10.0
    assert 0.0 <= alignment.backward_fraction <= 1.0
    assert alignment.route_length_m > 0


def test_alignment_rejects_missing_route(synthetic_root):
    root, trip_id, _ = synthetic_root
    trip = io.load_trip(root / "student_test" / trip_id, with_route=False)
    try:
        labels.align_track_to_route(trip)
    except ValueError:
        return
    raise AssertionError("alignment should require a reference route")
