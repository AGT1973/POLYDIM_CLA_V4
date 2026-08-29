import numpy as np
import pytest
from scientific_suite.reference.qr_oracle import controlled_matrix, orth_err, qr_residual

pytestmark = [pytest.mark.qr]


@pytest.mark.p0
def test_qr_has_both_orthogonality_and_residual(polydim, rng):
    Rng = rng
    A = Rng.standard_normal((128, 4))
    Q = np.asarray(polydim.CliffordRotors.cholesky_qr3(A))
    assert orth_err(Q) < 1e-10
    # Even without returned R, the best least-squares R under this Q should
    # reconstruct A. A unitary-looking Q that spans the wrong space must fail.
    assert qr_residual(A, Q) < 1e-8


@pytest.mark.p0
def test_qr_api_exposes_rank_or_breakdown_status(polydim):
    import inspect
    sig = inspect.signature(polydim.CliffordRotors.cholesky_qr3)
    doc = inspect.getdoc(polydim.CliffordRotors.cholesky_qr3) or ""
    source = __import__("pathlib").Path(polydim.__file__).read_text(encoding="utf-8")
    has_status = any(token in source for token in ["RANK_DEFICIENT", "rank_deficient", "status", "breakdown"])
    assert has_status, "QR API has no explicit rank/breakdown contract; positive shift can hide singular inputs"


@pytest.mark.p1
def test_qr_condition_sweep(polydim):
    for cond in [1e0, 1e4, 1e8, 1e12, 1e15, 1e16]:
        A = controlled_matrix(256, 4, cond, seed=int(np.log10(cond) * 13 + 1))
        Q = np.asarray(polydim.CliffordRotors.cholesky_qr3(A))
        assert np.all(np.isfinite(Q)), f"non-finite Q at cond={cond:g}"
        oe = orth_err(Q)
        re = qr_residual(A, Q)
        assert oe < 1e-8 or cond >= 1e15, (cond, oe)
        assert re < 1e-8 or cond >= 1e15, (cond, re)
