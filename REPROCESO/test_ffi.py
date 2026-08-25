import ctypes
import os

print("\n>>> ATAQUE 7: Hot-Reload C++ y Rust DLLs (E56 y E57 mitigados)")
try:
    cpp_dll = ctypes.CDLL(os.path.abspath("polydim_cpp_kernel.dll"))
    rust_dll = ctypes.CDLL(os.path.abspath("polydim_rust_kernel.dll"))
    print("  [OK] C++ AVX-512 DLL cargada.")
    print("  [OK] Rust FFI C-ABI DLL cargada.")
    
    # Test SeqLock en Rust
    read_begin = rust_dll.pmtp_seqlock_read_begin
    read_begin.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    read_begin.restype = ctypes.c_uint64
    
    seq_word = ctypes.c_uint64(0)
    seq_val = read_begin(ctypes.byref(seq_word))
    print(f"  [OK] SeqLock HW Barrier ejecutado via Rust FFI. seq_word={seq_val}")
    
    # Test C++ (si exporta función de prueba)
except Exception as e:
    print(f"  [X] FALLO FFI NATIVO: {e}")
