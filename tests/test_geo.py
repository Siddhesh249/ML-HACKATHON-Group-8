import numpy as np

from pathsnap import geo


def test_projection_roundtrip_and_scale():
    lonlat = np.array([[8.0, 49.0], [8.001, 49.0], [8.0, 49.001]])
    xy, origin = geo.project_lonlat(lonlat)
    back = geo.unproject_xy(xy, origin)
    assert np.allclose(back, lonlat, atol=1e-10)
    # 0.001 deg longitude at 49N is ~73 m; latitude is ~110.5 m.
    d_lon = float(np.linalg.norm(xy[1] - xy[0]))
    d_lat = float(np.linalg.norm(xy[2] - xy[0]))
    assert 70 < d_lon < 76
    assert 108 < d_lat < 113


def test_haversine_matches_projection_within_tolerance():
    lon1, lat1 = 8.0, 49.0
    lon2, lat2 = 8.01, 49.005
    d_geo = geo.haversine_m(lon1, lat1, lon2, lat2)
    xy, _ = geo.project_lonlat(np.array([[lon1, lat1], [lon2, lat2]]))
    d_plane = float(np.linalg.norm(xy[1] - xy[0]))
    assert abs(d_geo - d_plane) / d_geo < 0.01


def test_point_segment_distance_endpoint_and_interior():
    points = np.array([[0.0, 0.0], [5.0, 3.0]])
    a = np.array([[0.0, 0.0]])
    b = np.array([[10.0, 0.0]])
    dist, ratio, proj = geo.point_segment_distance(points, a, b)
    assert np.allclose(dist[0], [0.0])
    assert np.allclose(ratio[0], [0.0])
    assert np.allclose(dist[1], [3.0])
    assert np.allclose(ratio[1], [0.5])


def test_bearing_and_wrap():
    north = geo.bearing_deg(8.0, 49.0, 8.0, 49.1)
    east = geo.bearing_deg(8.0, 49.0, 8.1, 49.0)
    assert np.isclose(float(north), 0.0, atol=0.5)
    assert np.isclose(float(east), 90.0, atol=0.5)
    assert np.isclose(float(geo.angle_difference_deg(359.0, 1.0)), -2.0)


def test_smoothing_preserves_shape_and_handles_edges():
    lonlat = np.array([[8.0, 49.0], [8.0001, 49.0], [8.0002, 49.0]])
    smooth, heading, speed = geo.smooth_track(lonlat, np.array([1.0, 2.0, 3.0]), window=5)
    assert smooth.shape == lonlat.shape
    assert heading.shape == (3,)
    assert speed.shape == (3,)
    assert np.isfinite(heading).all()
    assert (speed >= 0).all()
