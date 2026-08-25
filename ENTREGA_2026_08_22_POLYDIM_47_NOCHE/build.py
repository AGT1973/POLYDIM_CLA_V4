# build.py
# Script Estándar de Compilación Nativa para DLLs (C++ y Rust) - POLYDIM V47
# ============================================================================

import os
import sys
import subprocess
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def ensure_source_files():
    cpp_src = os.path.join(BASE_DIR, "slerp_kernel_v47.cpp")
    rs_src = os.path.join(BASE_DIR, "lib_v47.rs")

    # Si falta el fuente C++, intenta extraerlo del monolito si está cargado
    if not os.path.exists(cpp_src):
        try:
            import polydim_v47_monolito
            if hasattr(polydim_v47_monolito, 'SLERP_KERNEL_CPP_SOURCE'):
                with open(cpp_src, 'w', encoding='utf-8') as f:
                    f.write(polydim_v47_monolito.SLERP_KERNEL_CPP_SOURCE)
                print(f"[BUILD] Extraído slerp_kernel_v47.cpp desde el monolito")
        except Exception:
            pass

    # Si falta el fuente Rust, intenta extraerlo del monolito si está cargado
    if not os.path.exists(rs_src):
        try:
            import polydim_v47_monolito
            if hasattr(polydim_v47_monolito, 'LIB_V47_RUST_SOURCE'):
                with open(rs_src, 'w', encoding='utf-8') as f:
                    f.write(polydim_v47_monolito.LIB_V47_RUST_SOURCE)
                print(f"[BUILD] Extraído lib_v47.rs desde el monolito")
        except Exception:
            pass

def find_msvc_vcvars() -> str:
    # 1. Intentar vswhere.exe estándar de Microsoft
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if os.path.exists(vswhere):
        try:
            res = subprocess.run([vswhere, "-latest", "-property", "installationPath"], capture_output=True, text=True)
            path = res.stdout.strip()
            if path:
                bat = os.path.join(path, "VC", "Auxiliary", "Build", "vcvars64.bat")
                if os.path.exists(bat):
                    return bat
        except Exception:
            pass

    # 2. Rutas conocidas de BuildTools y Visual Studio
    known_paths = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for p in known_paths:
        if os.path.exists(p):
            return p
    return ""

def compile_cpp_dll():
    print("[BUILD] Compilando kernel C++ (slerp_kernel_v47.cpp)...")
    ensure_source_files()
    cpp_src = os.path.join(BASE_DIR, "slerp_kernel_v47.cpp")
    cpp_dll = os.path.join(BASE_DIR, "slerp_kernel_v47.dll" if sys.platform == "win32" else "slerp_kernel_v47.so")

    if not os.path.exists(cpp_src):
        print(f"  -> ADVERTENCIA: No se encontró {cpp_src}. Se utilizará el motor de fallback JAX/NumPy.")
        return None

    # 1. Compilación MSVC en Windows
    vcvars = find_msvc_vcvars()
    if sys.platform == "win32" and vcvars:
        cmd = f'call "{vcvars}" && cl.exe /O2 /LD /std:c++17 "{cpp_src}" /Fe:"{cpp_dll}"'
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(cpp_dll):
            print(f"  -> C++ DLL compilada exitosamente con MSVC: {cpp_dll}")
            return cpp_dll

    # 2. Compilación g++ / clang++
    gpp = shutil.which("g++") or shutil.which("clang++")
    if gpp:
        flags = ["-O3", "-shared", "-std=c++17", "-fPIC", cpp_src, "-o", cpp_dll]
        res = subprocess.run([gpp] + flags, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(cpp_dll):
            print(f"  -> C++ DLL compilada exitosamente con GCC/Clang: {cpp_dll}")
            return cpp_dll

    print("  -> ADVERTENCIA: No se encontró compilador C++ (MSVC/g++). El motor usará fallback NumPy/JAX.")
    return None

def compile_rust_dll():
    print("[BUILD] Compilando módulo Rust Lock-Free MPMC (lib_v47.rs)...")
    ensure_source_files()
    rs_src = os.path.join(BASE_DIR, "lib_v47.rs")
    rs_dll = os.path.join(BASE_DIR, "lib_v47.dll" if sys.platform == "win32" else "lib_v47.so")

    if not os.path.exists(rs_src):
        print(f"  -> ADVERTENCIA: No se encontró {rs_src}. Se utilizará fallback Python threading.")
        return None

    rustc = shutil.which("rustc") or r"C:\Users\eluithi\.cargo\bin\rustc.EXE"
    if os.path.exists(rustc):
        flags = ["--crate-type=cdylib", "-C", "opt-level=3", rs_src, "-o", rs_dll]
        res = subprocess.run([rustc] + flags, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(rs_dll):
            print(f"  -> Rust DLL compilada exitosamente con rustc: {rs_dll}")
            return rs_dll

    print("  -> ADVERTENCIA: No se encontró rustc. El módulo MPMC usará fallback Python threading.")
    return None

def main():
    print("=== INICIANDO BUILD SYSTEM NATIVO POLYDIM V47 ===")
    compile_cpp_dll()
    compile_rust_dll()
    print("=== BUILD COMPLETADO ===")

if __name__ == "__main__":
    main()
