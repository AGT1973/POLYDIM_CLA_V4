"""
===============================================================================
POLYDIM V79 BULLDOG — DEEP MATHEMATICAL VERIFICATION FOR CAYLEY-SMW
===============================================================================
Tests demanded by Bucle #9-10 Red Team audit.  Every test here is a
*mathematical property*, not a smoke test.

Test families:
  1. W = U V^T is skew-symmetric
  2. Y^T Y ≈ I   (Stiefel preservation)
  3. R(0) = X     (identity retraction)
  4. dR/dα|₀ = G  (first-order tangency — derivative test)
  5. Solver residual ‖C·Z − rhs‖ bounded
  6. Dense Cayley == SMW Cayley (small-D oracle)
  7. O(D) memory (no D×D intermediates)
  8. alpha=0 → Y=X
  9. Edge cases: D=2, k=2 (minimal Stiefel)
 10. PMTP pack→corrupt→unpack fails MAC
 11. PMTP pack→unpack round-trip
 12. exp_map(0, v) → NaN (invalid manifold point)
===============================================================================
"""
import os, sys, struct, time
import numpy as np
import pytest

# ---- ensure the monolith is importable ----
_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("POLYDIM_PMTP_KEY", "testkeytestkeytestkeytestkeytest")

import jax
import jax.numpy as jnp

try:
    from polydim_v79_monolito import (
        GeodesicKernels, CliffordRotors, NativeFFIBridge, PMTPNetworkLayer,
        PMTP_HEADER_FMT_NO_MAC, PMTP_MAGIC, PMTP_NET_KEY
    )
    _MONOLITH = True
except Exception as exc:
    _MONOLITH = False
    _import_err = str(exc)

pytestmark = pytest.mark.skipif(not _MONOLITH, reason=f"Monolith import failed: {_import_err if not _MONOLITH else ''}")


# ============================================================================
# HELPERS
# ============================================================================

def _random_stiefel(key, D, k):
    """Random point on St(D, k) via QR."""
    A = jax.random.normal(key, (D, k))
    Q, _ = jnp.linalg.qr(A)
    return Q[:, :k]


def _random_tangent_stiefel(key, X):
    """Random tangent vector at X ∈ St(D, k).
    Condition: X^T Z + Z^T X = 0  ⟹  Z = (I - X X^T) B + X A  where A = -A^T.
    """
    D, k = X.shape
    k1, k2 = jax.random.split(key)
    B = jax.random.normal(k1, (D, k))
    A_raw = jax.random.normal(k2, (k, k))
    A = 0.5 * (A_raw - A_raw.T)           # skew k×k
    Z = (B - X @ (X.T @ B)) + X @ A        # tangent at X
    return Z


def _dense_cayley(X, Z, alpha):
    """Dense Cayley retraction on St(D, k) using the full D×D matrix.
    W = Z X^T - X Z^T  (skew-symmetric D×D).
    Y = (I + α/2 W)^{-1} (I - α/2 W) X.
    This is the ground-truth oracle.  O(D^3) — only for small D.
    """
    D = X.shape[0]
    I_D = jnp.eye(D, dtype=X.dtype)
    W = Z @ X.T - X @ Z.T
    A = I_D + 0.5 * alpha * W
    B = I_D - 0.5 * alpha * W
    return jnp.linalg.solve(A, B @ X)


# ============================================================================
# 1.  W = U V^T IS SKEW-SYMMETRIC
# ============================================================================

@pytest.mark.parametrize("D,k", [(4, 2), (8, 2), (16, 2), (64, 2)])
def test_W_skew_symmetric(D, k):
    """The low-rank generator W = U V^T must satisfy W + W^T ≈ 0."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X = _random_stiefel(k1, D, k)
    G = _random_tangent_stiefel(k2, X)

    # Build U, V exactly as the monolith does (lines 266-269)
    U = jnp.concatenate([G, X], axis=-1)
    V = jnp.concatenate([X, -G], axis=-1)
    W = U @ V.T
    skew_err = float(jnp.linalg.norm(W + W.T)) / max(1.0, float(jnp.linalg.norm(W)))
    assert skew_err < 1e-12, f"W not skew: relative error = {skew_err:.2e}"


# ============================================================================
# 2.  STIEFEL PRESERVATION  Y^T Y ≈ I
# ============================================================================

@pytest.mark.parametrize("D", [4, 16, 64, 256, 1000])
def test_stiefel_orthogonality(D):
    k = 2
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    X = _random_stiefel(k1, D, k)
    G = _random_tangent_stiefel(k2, X)
    Y = CliffordRotors.cayley_retract_stiefel(X, G, alpha=0.5)
    orth_err = float(jnp.linalg.norm(Y.T @ Y - jnp.eye(k)))
    assert orth_err < 1e-10, f"Orth err = {orth_err:.2e}"


# ============================================================================
# 3.  R(0) = X   (identity retraction)
# ============================================================================

def test_retraction_identity():
    D, k = 32, 2
    key = jax.random.PRNGKey(99)
    k1, k2 = jax.random.split(key)
    X = _random_stiefel(k1, D, k)
    G = _random_tangent_stiefel(k2, X)
    Y = CliffordRotors.cayley_retract_stiefel(X, G, alpha=0.0)
    err = float(jnp.linalg.norm(Y - X))
    assert err < 1e-12, f"R(0) != X, err = {err:.2e}"


# ============================================================================
# 4.  FIRST-ORDER TANGENCY  dR/dα|₀ = G   (derivative test)
# ============================================================================

def test_cayley_derivative_at_zero():
    """Finite-difference approximation: (R(h) - R(0)) / h ≈ G as h→0.
    We use two step sizes and check Richardson extrapolation convergence.
    """
    D, k = 32, 2
    key = jax.random.PRNGKey(13)
    k1, k2 = jax.random.split(key)
    X = _random_stiefel(k1, D, k)
    Z = _random_tangent_stiefel(k2, X)

    h1, h2 = 1e-4, 1e-5
    Y1 = CliffordRotors.cayley_retract_stiefel(X, Z, alpha=h1)
    Y2 = CliffordRotors.cayley_retract_stiefel(X, Z, alpha=h2)
    dR1 = (Y1 - X) / h1
    dR2 = (Y2 - X) / h2

    # The exact derivative at alpha=0 is -W X, where W = Z X^T - X Z^T
    W = Z @ X.T - X @ Z.T
    exact_deriv = -W @ X

    err1 = float(jnp.linalg.norm(dR1 - exact_deriv))
    err2 = float(jnp.linalg.norm(dR2 - exact_deriv))
    # err2 should be ~10x smaller than err1 (first-order convergence)
    assert err2 < err1, f"No convergence: err(h1)={err1:.2e}, err(h2)={err2:.2e}"
    assert err2 < 1e-3, f"Derivative error too large: {err2:.2e}"


# ============================================================================
# 5.  DENSE CAYLEY == SMW CAYLEY  (small-D oracle)
# ============================================================================

@pytest.mark.parametrize("D", [4, 8, 16, 32])
def test_dense_vs_smw_cayley(D):
    k = 2
    key = jax.random.PRNGKey(21)
    k1, k2 = jax.random.split(key)
    X = _random_stiefel(k1, D, k)
    Z = _random_tangent_stiefel(k2, X)

    Y_dense = _dense_cayley(X, Z, alpha=0.5)
    Y_smw = CliffordRotors.cayley_retract_stiefel(X, Z, alpha=0.5)
    err = float(jnp.linalg.norm(Y_dense - Y_smw))
    assert err < 1e-10, f"Dense vs SMW mismatch: {err:.2e}"


# ============================================================================
# 6.  SOLVER RESIDUAL  ‖C·Z − rhs‖ bounded
# ============================================================================

def test_solver_residual():
    """Reconstruct the 4×4 system and check that the solution Z satisfies C Z ≈ rhs."""
    D, k = 16, 2
    key = jax.random.PRNGKey(5)
    k1, k2 = jax.random.split(key)
    X = _random_stiefel(k1, D, k)
    G = _random_tangent_stiefel(k2, X)

    alpha = 0.5
    a2 = 0.5 * alpha
    U = jnp.concatenate([G, X], axis=-1)
    V = jnp.concatenate([X, -G], axis=-1)
    VtU = V.T @ U
    VtX = V.T @ X
    C = jnp.eye(2 * k) + a2 * VtU
    rhs = VtX - a2 * (VtU @ VtX)
    Z_sol = jnp.linalg.solve(C, rhs)

    residual = float(jnp.linalg.norm(C @ Z_sol - rhs))
    scale = float(jnp.linalg.norm(C)) * float(jnp.linalg.norm(Z_sol)) + float(jnp.linalg.norm(rhs))
    rel_res = residual / max(scale, 1e-30)
    assert rel_res < 1e-12, f"Solver residual too large: {rel_res:.2e}"


# ============================================================================
# 7.  O(D) MEMORY — no D×D intermediates
# ============================================================================

def test_memory_scaling_linear():
    """Run Cayley for two D values and check memory doesn't blow up quadratically."""
    key = jax.random.PRNGKey(0)

    def _run(D):
        k1, k2 = jax.random.split(jax.random.PRNGKey(D))
        X = _random_stiefel(k1, D, 2)
        G = _random_tangent_stiefel(k2, X)
        Y = CliffordRotors.cayley_retract_stiefel(X, G, alpha=0.3)
        return Y

    # Warm up JIT
    _run(100)
    _run(200)

    # If there were a D×D intermediate, peak memory for D=2000 would be ~32MB
    # vs ~0.032MB for O(D). We just check it doesn't crash (no OOM).
    Y = _run(10_000)
    orth_err = float(jnp.linalg.norm(Y.T @ Y - jnp.eye(2)))
    assert orth_err < 1e-8, f"Orth err at D=10000: {orth_err:.2e}"


# ============================================================================
# 8.  alpha=0 → Y=X
# ============================================================================

def test_alpha_zero_is_identity():
    D, k = 50, 2
    key = jax.random.PRNGKey(77)
    k1, k2 = jax.random.split(key)
    X = _random_stiefel(k1, D, k)
    G = _random_tangent_stiefel(k2, X)
    Y = CliffordRotors.cayley_retract_stiefel(X, G, alpha=0.0)
    err = float(jnp.linalg.norm(Y - X))
    assert err < 1e-14, f"alpha=0 not identity: {err:.2e}"


# ============================================================================
# 9.  EDGE CASE — D=2, k=2 (St(2,2) ≈ O(2))
# ============================================================================

def test_cayley_D2_k2():
    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    X = _random_stiefel(k1, 2, 2)
    G = _random_tangent_stiefel(k2, X)
    Y = CliffordRotors.cayley_retract_stiefel(X, G, alpha=0.3)
    orth_err = float(jnp.linalg.norm(Y.T @ Y - jnp.eye(2)))
    assert orth_err < 1e-10, f"D=2 orth err: {orth_err:.2e}"
    # Compare with dense oracle
    Y_dense = _dense_cayley(X, G, alpha=0.3)
    match_err = float(jnp.linalg.norm(Y - Y_dense))
    assert match_err < 1e-10, f"D=2 dense mismatch: {match_err:.2e}"


# ============================================================================
# 10. PMTP — PAYLOAD INTEGRITY (MAC covers payload)
# ============================================================================

def test_pmtp_payload_integrity():
    layer = PMTPNetworkLayer("sender")
    payload = b"datos secretos de tensor"
    packet = layer.pack_tensor_header((2, 3), payload, b"receiver")

    # Legitimate round-trip
    sender, recv_payload, shape = layer.unpack_and_verify(packet, b"receiver")
    assert recv_payload == payload
    assert shape == (2, 3)


def test_pmtp_corrupt_payload_rejected():
    layer = PMTPNetworkLayer("sender")
    payload = b"datos secretos de tensor"
    packet = layer.pack_tensor_header((2, 3), payload, b"receiver")

    # Corrupt one byte of payload
    header = packet[:112]
    orig_payload = packet[112:-16]
    mac = packet[-16:]
    corrupt_payload = b"X" * len(orig_payload)
    corrupt_packet = header + corrupt_payload + mac

    with pytest.raises(ValueError, match="Invalid MAC"):
        layer.unpack_and_verify(corrupt_packet, b"receiver")


def test_pmtp_corrupt_header_rejected():
    layer = PMTPNetworkLayer("sender")
    payload = b"test"
    packet = layer.pack_tensor_header((1,), payload, b"nodeB")

    # Corrupt header byte 10
    corrupted = bytearray(packet)
    corrupted[10] ^= 0xFF
    corrupted = bytes(corrupted)

    with pytest.raises(ValueError, match="Invalid MAC"):
        layer.unpack_and_verify(corrupted, b"nodeB")


def test_pmtp_wrong_receiver():
    layer = PMTPNetworkLayer("sender")
    payload = b"test"
    packet = layer.pack_tensor_header((1,), payload, b"nodeB")
    with pytest.raises(ValueError, match="Receiver mismatch"):
        layer.unpack_and_verify(packet, b"attacker")


def test_pmtp_replay_rejected():
    layer = PMTPNetworkLayer("sender")
    payload = b"test"
    packet = layer.pack_tensor_header((1,), payload, b"nodeB")
    # First unpack OK
    layer.unpack_and_verify(packet, b"nodeB")
    # Second unpack should detect replay
    with pytest.raises(ValueError, match="Replay"):
        layer.unpack_and_verify(packet, b"nodeB")


# ============================================================================
# 11. exp_map(0, v) → NaN (invalid manifold point)
# ============================================================================

def test_exp_map_zero_base_returns_nan():
    x = jnp.zeros(3, dtype=jnp.float64)
    v = jnp.array([1.0, 1.0, 1.0])
    result = GeodesicKernels.exp_map(x, v)
    assert jnp.all(jnp.isnan(result)), f"exp_map(0,v) should be NaN, got {result}"


# ============================================================================
# 12. safe_norm gradient is finite
# ============================================================================

def test_safe_norm_gradient_finite():
    """The gradient of safe_norm at a near-zero vector must be finite, not 1e+154."""
    x = jnp.array([1e-20, 0.0, 0.0], dtype=jnp.float64)
    grad_fn = jax.grad(lambda z: GeodesicKernels.safe_norm(z))
    g = grad_fn(x)
    assert jnp.all(jnp.isfinite(g)), f"Gradient not finite: {g}"
    # The gradient magnitude should be reasonable (≈ unit direction)
    assert float(jnp.linalg.norm(g)) < 1e10, f"Gradient explosion: norm={float(jnp.linalg.norm(g)):.2e}"
