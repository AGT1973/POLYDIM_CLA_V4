import ctypes
import numpy as np
import os

cpp_lib = ctypes.CDLL(os.path.abspath("./slerp_kernel_v47_matrix_free.dll"))
cpp_lib.polydim_slerp_native_nd.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double)
]
cpp_lib.polydim_slerp_native_nd.restype = ctypes.c_int

np.random.seed(42)
q1 = np.random.randn(500); q1 /= np.linalg.norm(q1)
q2 = -q1.copy()

D = len(q1)
out_cpp = np.zeros(D, dtype=np.float64)
q1_ptr = q1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
q2_ptr = q2.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

res = cpp_lib.polydim_slerp_native_nd(q1_ptr, q2_ptr, 0.5, D, out_cpp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
print(f"Res: {res}")
print(f"Out: {out_cpp[:5]}")
print(f"Has NaN: {np.isnan(out_cpp).any()}")
