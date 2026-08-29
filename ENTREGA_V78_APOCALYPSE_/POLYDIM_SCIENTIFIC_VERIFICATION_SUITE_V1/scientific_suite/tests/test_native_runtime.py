from __future__ import annotations
import ctypes
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from scientific_suite.reference.sphere_oracle import householder, stiefel_full_cayley

pytestmark = [pytest.mark.ffi]


@pytest.fixture(scope="session")
def cpp_lib(tmp_path_factory):
    gxx = shutil.which("g++") or shutil.which("clang++")
    if not gxx:
        pytest.skip("No C++ compiler available")
    suite_root = Path(__file__).resolve().parents[1]
    project = Path(__import__("os").environ.get("POLYDIM_PROJECT", suite_root.parent))
    src = project / "kernel_cpp_v78.cpp"
    if not src.exists():
        alt = project / "kernel_cpp_v78.cpp.txt"
        if alt.exists():
            src = alt
        else:
            pytest.skip("C++ source not found")
    outdir = tmp_path_factory.mktemp("native")
    out = outdir / "libpolydim_v78.so"
    compile_src = outdir / "kernel.cpp"
    compile_src.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    subprocess.run([gxx, "-O2", "-std=c++20", "-fPIC", "-shared", str(compile_src), "-o", str(out)], check=True)
    lib = ctypes.CDLL(str(out))
    hh = lib.polydim_householder_reflect_cpp
    hh.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_uint64]
    hh.restype = ctypes.c_int
    cayley = lib.polydim_cayley_retract_k2_cpp
    cayley.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_uint64, ctypes.c_double]
    cayley.restype = ctypes.c_int
    return lib


def as_ptr(a):
    return np.ascontiguousarray(a, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))


@pytest.mark.p0
def test_cpp_householder_matches_oracle(cpp_lib):
    rng = np.random.default_rng(4)
    x = rng.standard_normal(64)
    v = rng.standard_normal(64)
    out = np.empty_like(x)
    code = cpp_lib.polydim_householder_reflect_cpp(as_ptr(x), as_ptr(v), as_ptr(out), len(x))
    assert code == 0
    ref = householder(x, v)
    assert np.linalg.norm(out - ref) / np.linalg.norm(ref) < 1e-12


@pytest.mark.p0
def test_cpp_householder_aliasing_is_rejected_or_handled_correctly(cpp_lib):
    x = np.arange(16, dtype=np.float64)
    v = np.linspace(1.0, 2.0, 16)
    # Exact out == x is a valid in-place possibility only if explicitly supported.
    before = x.copy()
    code = cpp_lib.polydim_householder_reflect_cpp(as_ptr(x), as_ptr(v), as_ptr(x), len(x))
    ref = householder(before, v)
    assert code != 0 or np.linalg.norm(x - ref) < 1e-12, \
        "C++ claimed success for x/out aliasing without a correct in-place result"


@pytest.mark.p0
def test_cpp_cayley_is_not_a_linear_euler_stub(cpp_lib):
    rng = np.random.default_rng(9)
    A = rng.standard_normal((12, 2))
    X, _ = np.linalg.qr(A)
    G = rng.standard_normal((12, 2))
    out = np.empty_like(X)
    alpha = 0.4
    code = cpp_lib.polydim_cayley_retract_k2_cpp(as_ptr(X), as_ptr(G), as_ptr(out), 12, alpha)
    assert code == 0
    ref = stiefel_full_cayley(X, G, alpha)
    assert np.linalg.norm(out - ref) / np.linalg.norm(ref) < 1e-9, \
        "Native Cayley output does not match the mathematically defined Cayley transform"


@pytest.mark.p0
def test_cpp_cayley_preserves_stiefel_constraint(cpp_lib):
    rng = np.random.default_rng(10)
    X, _ = np.linalg.qr(rng.standard_normal((17, 2)))
    G = rng.standard_normal((17, 2))
    out = np.empty_like(X)
    code = cpp_lib.polydim_cayley_retract_k2_cpp(as_ptr(X), as_ptr(G), as_ptr(out), 17, 0.7)
    assert code == 0
    assert np.linalg.norm(out.T @ out - np.eye(2)) < 1e-10
