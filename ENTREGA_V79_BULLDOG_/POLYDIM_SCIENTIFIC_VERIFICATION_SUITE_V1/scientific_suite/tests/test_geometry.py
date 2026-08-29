import math
import numpy as np
import pytest
from scientific_suite.reference.sphere_oracle import exp_map as ref_exp, log_map as ref_log

pytestmark = [pytest.mark.geometry]


def unit_pair(d, theta, rng):
    x = rng.standard_normal(d)
    x /= np.linalg.norm(x)
    u = rng.standard_normal(d)
    u -= np.dot(u, x) * x
    u /= np.linalg.norm(u)
    y = math.cos(theta) * x + math.sin(theta) * u
    return x, u, y


@pytest.mark.p0
def test_reference_exp_log_identity():
    rng = np.random.default_rng(7)
    for theta in [0.0, 1e-12, 1e-8, 1e-4, 0.3, math.pi / 2, math.pi - 1e-4, math.pi - 1e-8]:
        x, _, y = unit_pair(17, theta, rng)
        if theta > 0 and abs(math.pi - theta) > 1e-11:
            v = ref_log(x, y)
            y2 = ref_exp(x, v)
            assert np.linalg.norm(y2 - y) < 2e-12


@pytest.mark.p0
def test_target_rejects_zero_manifold_point(polydim):
    G = polydim.GeodesicKernels
    x = np.zeros(8)
    y = np.zeros(8); y[0] = 1
    # A scientific API must not silently reinterpret zero as a sphere point.
    out = np.asarray(G.log_map(x, y))
    assert not np.all(np.isfinite(out)) or np.linalg.norm(out) > 1e-3, \
        "V78 silently maps zero input into a plausible geometric result"


@pytest.mark.p0
def test_target_antipodal_is_not_claimed_as_smooth(polydim):
    G = polydim.GeodesicKernels
    x = np.zeros(8); x[0] = 1
    y = -x
    out = np.asarray(G.log_map(x, y))
    # The correct mathematical contract must either return an explicit branch
    # with documented status or refuse the multi-valued point.
    ref = None
    try:
        ref = ref_log(x, y)
    except ValueError:
        pass
    assert ref is None
    # V78 should not present an arbitrary finite vector as a uniquely defined log.
    assert np.linalg.norm(out) == 0 or not np.all(np.isfinite(out)), \
        "Antipodal Log returned a finite non-zero vector as if the branch were unique"


@pytest.mark.p1
@pytest.mark.parametrize("theta", [1e-14, 1e-12, 1e-8, 1e-4, 1e-2])
def test_target_roundtrip_controlled_angles(polydim, rng, theta):
    G = polydim.GeodesicKernels
    x, _, y = unit_pair(64, theta, rng)
    v = np.asarray(G.log_map(x, y))
    y2 = np.asarray(G.exp_map(x, v))
    err = np.linalg.norm(y2 - y)
    assert err < 1e-6, f"theta={theta:g}, err={err:.3e}"


@pytest.mark.p1
def test_exp_requires_unit_basepoint(polydim):
    G = polydim.GeodesicKernels
    x = np.array([2.0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    v = np.array([0.0, 0.3, 0, 0, 0, 0, 0, 0], dtype=float)
    out = np.asarray(G.exp_map(x, v))
    # Contract test: a sphere Exp must either validate/normalize explicitly.
    expected_if_validated = np.array([math.cos(0.3), math.sin(0.3), 0, 0, 0, 0, 0, 0])
    assert np.linalg.norm(out - expected_if_validated) < 1e-8 or np.linalg.norm(out - x/np.linalg.norm(x)) < 1e-8, \
        "Exp silently interpreted a non-unit basepoint inconsistently"


@pytest.mark.p1
def test_newton_is_not_needed_for_closed_form_and_should_not_worsen(polydim, rng):
    G = polydim.GeodesicKernels
    x, _, y = unit_pair(32, 0.7, rng)
    v_ref = np.asarray(G.log_map(x, y))
    # Perturb the closed-form solution; a correct Newton implementation should
    # reduce the Exp(Log) residual, not merely perform arbitrary tangent updates.
    perturb = np.zeros_like(v_ref); perturb[0] = 0.05
    if abs(np.dot(perturb, x)) > 0:
        perturb -= np.dot(perturb, x) * x
    v0 = v_ref + perturb
    e0 = np.linalg.norm(np.asarray(G.exp_map(x, v0)) - y)
    v1 = np.asarray(G.log_map_newton(x, y, max_iter=5))
    e1 = np.linalg.norm(np.asarray(G.exp_map(x, v1)) - y)
    assert e1 <= e0 + 1e-10, f"Newton worsened residual: {e0:.3e} -> {e1:.3e}"
