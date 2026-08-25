import os
import subprocess
import shutil

dir_v47 = r"E:\POLYDIM_EINSOF\ENTREGA_2026_08_22_POLYDIM_47_NOCHE"
cpp_src = os.path.join(dir_v47, "slerp_kernel_v47.cpp")
cpp_dll = os.path.join(dir_v47, "slerp_kernel_v47.dll")

vcvars = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if os.path.exists(vcvars):
    cmd = f'call "{vcvars}" && cl.exe /O2 /LD /std:c++17 "{cpp_src}" /Fe:"{cpp_dll}"'
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("C++ MSVC Compilation Code:", res.returncode)
    print("C++ DLL exists:", os.path.exists(cpp_dll))
    if not os.path.exists(cpp_dll):
        print("MSVC Output:", res.stdout)
        print("MSVC Errors:", res.stderr)

rustc = r"C:\Users\eluithi\.cargo\bin\rustc.EXE"
rs_src = os.path.join(dir_v47, "lib_v47.rs")
rs_dll = os.path.join(dir_v47, "lib_v47.dll")

if os.path.exists(rustc):
    res_rs = subprocess.run([rustc, "--crate-type=cdylib", "-C", "opt-level=3", rs_src, "-o", rs_dll], capture_output=True, text=True)
    print("Rust Compilation Code:", res_rs.returncode)
    print("Rust DLL exists:", os.path.exists(rs_dll))
    if not os.path.exists(rs_dll):
        print("Rustc Stdout:", res_rs.stdout)
        print("Rustc Stderr:", res_rs.stderr)
