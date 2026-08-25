import ctypes
import numpy as np
import sys
import os

def load_libs():
    try:
        rust_lib = ctypes.CDLL(os.path.abspath("./lib_v47_matrix_free.dll"))
        cpp_lib = ctypes.CDLL(os.path.abspath("./slerp_kernel_v47_matrix_free.dll"))
    except Exception as e:
        print(f"Error loading DLLs: {e}")
        sys.exit(1)

    rust_lib.polydim_rust_slerp_nd.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double)
    ]
    rust_lib.polydim_rust_slerp_nd.restype = ctypes.c_int

    cpp_lib.polydim_slerp_native_nd.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double)
    ]
    cpp_lib.polydim_slerp_native_nd.restype = ctypes.c_int

    return rust_lib, cpp_lib

def py_slerp(q1, q2, t):
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    if dot <= -1.0 + 1e-12: # Antipodal fallback matching C++ / Rust
        idx = np.argmin(np.abs(q1))
        e = np.zeros_like(q1)
        e[idx] = 1.0
        r = e - q1[idx] * q1
        r /= np.linalg.norm(r)
        theta = np.pi * t
        return np.cos(theta) * q1 + np.sin(theta) * r

    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    if sin_omega < 1e-10:
        return (1-t) * q1 + t * q2
    
    return (np.sin((1-t)*omega) / sin_omega) * q1 + (np.sin(t*omega) / sin_omega) * q2

def test_slerp(rust_lib, cpp_lib, name, q1, q2, t):
    D = len(q1)
    out_rust = np.zeros(D, dtype=np.float64)
    out_cpp = np.zeros(D, dtype=np.float64)

    q1_ptr = q1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    q2_ptr = q2.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    rust_res = rust_lib.polydim_rust_slerp_nd(q1_ptr, q2_ptr, t, D, out_rust.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    cpp_res = cpp_lib.polydim_slerp_native_nd(q1_ptr, q2_ptr, t, D, out_cpp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    out_py = py_slerp(q1, q2, t)
    out_py /= np.linalg.norm(out_py)
    
    err_cpp = np.linalg.norm(out_py - out_cpp)
    err_rust = np.linalg.norm(out_py - out_rust)
    err_cross = np.linalg.norm(out_cpp - out_rust)

    print(f"[{name}] D={D} t={t}")
    print(f"  Py vs C++ error:  {err_cpp:.2e} (cpp_res={cpp_res})")
    print(f"  Py vs Rust error: {err_rust:.2e} (rust_res={rust_res})")
    print(f"  C++ vs Rust:      {err_cross:.2e}")
    
    assert err_cpp < 1e-10, f"C++ mismatch on {name}"
    assert err_rust < 1e-10, f"Rust mismatch on {name}"
    assert err_cross < 1e-10, f"Cross mismatch on {name}"

if __name__ == "__main__":
    rust_lib, cpp_lib = load_libs()
    
    print("=== CROSS-BACKEND EQUIVALENCE TEST ===")
    
    # 1. Normal case
    np.random.seed(42)
    q1 = np.random.randn(100); q1 /= np.linalg.norm(q1)
    q2 = np.random.randn(100); q2 /= np.linalg.norm(q2)
    test_slerp(rust_lib, cpp_lib, "Normal", q1, q2, 0.3)
    
    # 2. Antipodal case
    q1 = np.random.randn(500); q1 /= np.linalg.norm(q1)
    q2 = -q1.copy()
    test_slerp(rust_lib, cpp_lib, "Antipodal", q1, q2, 0.5)
    
    # 3. Degenerate (Orthogonal)
    q1 = np.zeros(200); q1[0] = 1.0
    q2 = np.zeros(200); q2[1] = 1.0
    test_slerp(rust_lib, cpp_lib, "Orthogonal", q1, q2, 0.75)
    
    # 4. Identity
    q1 = np.random.randn(1000); q1 /= np.linalg.norm(q1)
    test_slerp(rust_lib, cpp_lib, "Identity", q1, q1, 0.1)

    print("ALL TESTS PASSED: 100% CROSS-BACKEND PARITY ESTABLISHED.")
