"""
Orquestador de Tests del Triple Núcleo POLYDIM EINSOF V40
Ejecuta automáticamente la suite de JAX, compila/testea Rust y compila/testea C++.
"""
import os, sys, subprocess, shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_step(name, cmd, cwd=BASE_DIR):
    print(f"\n{'-'*60}")
    print(f"🚀 INICIANDO: {name}")
    print(f"📂 CWD: {cwd}")
    print(f"💻 CMD: {cmd}")
    print(f"{'-'*60}\n")
    
    try:
        res = subprocess.run(cmd, cwd=cwd, shell=True)
        if res.returncode == 0:
            print(f"\n✅ [PASS] {name}")
            return True
        else:
            print(f"\n❌ [FAIL] {name} (Código de salida: {res.returncode})")
            return False
    except Exception as e:
        print(f"\n❌ [ERROR] Falló al lanzar {name}: {e}")
        return False

def main():
    print(r"""
  _____  ____  _ __   __ _____  _____ __  __ 
 |  __ \|  _ \| |\ \ / // ____||_   _|  \/  |
 | |__) | |_) | | \ V /| |       | | | \  / |
 |  ___/|  _ <| |  > < | |       | | | |\/| |
 | |    | |_) | | / . \| |____  _| |_| |  | |
 |_|    |____/|_|/_/ \_\\_____||_____|_|  |_|
 
    V40 - TEST SUITE TRIPLE NÚCLEO 
    (JAX + RUST + C++)
    """)
    
    all_success = True
    
    # ---------------------------------------------------------
    # 1. JAX (PYTHON)
    # ---------------------------------------------------------
    all_success &= run_step(
        "Capa JAX (Python)", 
        f'"{sys.executable}" test_jax.py', 
        cwd=os.path.join(BASE_DIR, "einsof_jax")
    )
    
    # ---------------------------------------------------------
    # 2. RUST (PMTP Bus)
    # ---------------------------------------------------------
    # Chequeamos si cargo existe
    if shutil.which("cargo"):
        os.environ["PYO3_USE_ABI3_FORWARD_COMPATIBILITY"] = "1"
        all_success &= run_step(
            "Capa PMTP (Rust)", 
            "cargo test --release", 
            cwd=os.path.join(BASE_DIR, "einsof_rust")
        )
    else:
        print("\n⚠️ 'cargo' NO ENCONTRADO en el PATH. Omitiendo tests de Rust.")
        all_success = False

    # ---------------------------------------------------------
    # 3. C++ (Kernel)
    # ---------------------------------------------------------
    cpp_dir = os.path.join(BASE_DIR, "einsof_cpp")
    build_dir = os.path.join(cpp_dir, "build")
    
    if shutil.which("cmake"):
        b1 = run_step("CMake Configure (C++)", "cmake -S . -B build", cwd=cpp_dir)
        b2 = False
        if b1:
            b2 = run_step("CMake Build (C++)", "cmake --build build --config Release", cwd=cpp_dir)
            
        if b2:
            # Buscar el ejecutable donde sea que CMake lo haya tirado
            exe_paths = [
                os.path.join(build_dir, "Release", "test_cpp.exe"),
                os.path.join(build_dir, "Debug", "test_cpp.exe"),
                os.path.join(build_dir, "test_cpp.exe"),
                os.path.join(build_dir, "test_cpp")
            ]
            
            exe_found = False
            for p in exe_paths:
                if os.path.exists(p):
                    all_success &= run_step("Test Ejecutable C++", f'"{p}"', cwd=build_dir)
                    exe_found = True
                    break
                    
            if not exe_found:
                print("\n❌ No se encontró el ejecutable test_cpp compilado.")
                all_success = False
        else:
            all_success = False
    else:
        print("\n⚠️ 'cmake' NO ENCONTRADO en el PATH. No se puede compilar C++ automáticamente aquí.")
        print("Abre este script desde una 'Developer Command Prompt' si estás en Windows.")
        all_success = False

    # ---------------------------------------------------------
    # RESULTADO
    # ---------------------------------------------------------
    print(f"\n{'='*60}")
    if all_success:
        print("🎉 RESULTADO FINAL: TODOS LOS NÚCLEOS OK (15/15 CHKs certificados)")
        sys.exit(0)
    else:
        print("⚠️ RESULTADO FINAL: ALGUNAS PRUEBAS FALLARON O NO PUDIERON COMPILARSE")
        sys.exit(1)

if __name__ == "__main__":
    main()
