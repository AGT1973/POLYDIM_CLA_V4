import numpy as np
import pytest
from scientific_suite.reference.sphere_oracle import stiefel_full_cayley, stiefel_smw_wenyin

pytestmark = [pytest.mark.cayley]


def random_stiefel(d, k, rng):
    A = rng.standard_normal((d, k))
    Q, _ = np.linalg.qr(A)
    return Q[:, :k]


@pytest.mark.p0
def test_full_cayley_equals_smw_oracle(rng):
    for d, k in [(5, 1), (8, 2), (11, 3)]:
        X = random_stiefel(d, k, rng)
        G = rng.standard_normal((d, k))
        Yfull = stiefel_full_cayley(X, G, 0.17)
        Ysmw = stiefel_smw_wenyin(X, G, 0.17)
        err = np.linalg.norm(Yfull - Ysmw)
        assert err < 1e-11, (d, k, err)


@pytest.mark.p0
def test_full_cayley_preserves_stiefel_constraint(rng):
    X = random_stiefel(13, 3, rng)
    G = rng.standard_normal((13, 3))
    Y = stiefel_full_cayley(X, G, 0.7)
    err = np.linalg.norm(Y.T @ Y - np.eye(3))
    assert err < 1e-11


@pytest.mark.p1
def test_cayley_target_does_not_reduce_to_linear_euler_step(polydim, rng):
    X = random_stiefel(32, 2, rng)
    G = rng.standard_normal((32, 2))
    alpha = 0.3
    reference = stiefel_full_cayley(X, G, alpha)
    # Current V78 native-like kernel shown in the delivery is X - .01 alpha G.
    euler_like = X - 0.01 * alpha * G
    err_ref = np.linalg.norm(reference - euler_like)
    assert err_ref > 1e-6
