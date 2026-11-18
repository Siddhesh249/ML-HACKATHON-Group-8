from pathsnap import io


def test_discover_and_load(synthetic_root):
    root, trip_id, route = synthetic_root
    metas = io.discover_trips(root, partition="student_test")
    assert len(metas) == 1
    assert metas[0].trip_id == trip_id
    assert metas[0].n_route_arcs == len(route)

    trip = io.load_trip(metas[0].directory, with_route=True)
    assert trip.route is not None
    assert trip.route.tolist() == route.tolist()
    assert trip.track.lonlat.shape[1] == 2


def test_prediction_load_has_no_route(synthetic_root):
    root, trip_id, _ = synthetic_root
    trip = io.load_trip(root / "student_test" / trip_id, with_route=False)
    assert trip.route is None


def test_resolve_trip_dir(synthetic_root, monkeypatch):
    root, trip_id, _ = synthetic_root
    import pathsnap.config as config

    monkeypatch.setattr(config, "DATA_ROOT", root)
    path = io.resolve_trip_dir(trip_id, root)
    assert path.name == trip_id
    assert path.parent.name == "student_test"


def test_invalid_trip_id_rejected(synthetic_root):
    root, _, _ = synthetic_root
    try:
        io.resolve_trip_dir("../etc", root)
    except io.DataError:
        return
    raise AssertionError("path traversal was not rejected")
