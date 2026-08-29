import numpy as np
import pytest
from scientific_suite.reference.sphere_oracle import householder

pytestmark = [pytest.mark.householder]


def call_target(polydim, x, v):
    return np.asarray(polydim.NativeFFIBridge.householder_reflect(x, v))


@pytest.mark.p0
def test_householder_matches_reference(polydim, rng):
    x = rng.standard_normal(64)
    v = rng.standard_normal(64)
    out = call_target(polydim, x, v)
    ref = householder(x, v)
    assert np.linalg.norm(out - ref) / np.linalg.norm(ref) < 1e-12


@pytest.mark.p0
@pytest.mark.parametrize("scale", [1e-20, 1e-12, 1e-8, 1.0, 1e8, 1e20])
def test_householder_scale_invariance(polydim, rng, scale):
    x = rng.standard_normal(32)
    v = rng.standard_normal(32)
    out = call_target(polydim, x, v * scale)
    ref = call_target(polydim, x, v)
    err = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    assert err < 1e-10, f"scale={scale:g}, relative error={err:.3e}"


@pytest.mark.p0
def test_householder_involution(polydim, rng):
    x = rng.standard_normal(64)
    v = rng.standard_normal(64)
    y = call_target(polydim, x, v)
    z = call_target(polydim, y, v)
    assert np.linalg.norm(z - x) / np.linalg.norm(x) < 1e-12


@pytest.mark.p0
def test_householder_norm_preservation(polydim, rng):
    x = rng.standard_normal(64)
    v = rng.standard_normal(64)
    y = call_target(polydim, x, v)
    assert abs(np.linalg.norm(y) - np.linalg.norm(x)) / np.linalg.norm(x) < 1e-12


@pytest.mark.p1
def test_zero_reflector_has_explicit_policy(polydim):
    x = np.arange(8, dtype=float)
    v = np.zeros(8)
    out = call_target(polydim, x, v)
    # Identity is acceptable only if this is an explicit contract for v=0;
    # V78 must not confuse small-norm nonzero vectors with zero.
    assert np.allclose(out, x)


@pytest.mark.p1
def test_nan_is_not_sanitized_to_a_fake_valid_reflection(polydim):
    x = np.ones(8)
    v = np.ones(8); v[0] = np.nan
    out = call_target(polydim, x, v)
    assert not np.all(np.isfinite(out)), "NaN input was silently converted to plausible finite output"
