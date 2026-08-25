import re
import os
import subprocess

def main():
    print("Extracting native sources from polydim_v62_monolito.py...")
    with open('polydim_v62_monolito.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract CPP
    cpp_match = re.search(r'CPP_SOURCE = r"""(.*?)"""', content, re.DOTALL)
    if not cpp_match:
        print("ERROR: CPP_SOURCE no encontrado")
        return
    with open('polydim_cpp_kernel.cpp', 'w', encoding='utf-8') as f:
        f.write(cpp_match.group(1))

    # Extract RUST
    rust_match = re.search(r'RUST_SOURCE = r"""(.*?)"""', content, re.DOTALL)
    if not rust_match:
        print("ERROR: RUST_SOURCE no encontrado")
        return
    with open('polydim_rust_kernel.rs', 'w', encoding='utf-8') as f:
        f.write(rust_match.group(1))

    print("Compiling Rust Kernel...")
    res = subprocess.run(["rustc", "--crate-type", "cdylib", "polydim_rust_kernel.rs"])
    if res.returncode == 0:
        print("Rust DLL compiled successfully.")
    else:
        print("Rust DLL compilation failed.")
        
    print("Compiling C++ Kernel...")
    vcvars = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    cmd = f'"{vcvars}" && cl.exe /LD /EHsc polydim_cpp_kernel.cpp'
    res = subprocess.run(cmd, shell=True)
    if res.returncode == 0:
        print("C++ DLL compiled successfully.")
    else:
        print("C++ DLL compilation failed.")

if __name__ == "__main__":
    main()
