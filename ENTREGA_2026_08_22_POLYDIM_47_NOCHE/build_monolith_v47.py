import os

dir_v47 = r"E:\POLYDIM_EINSOF\ENTREGA_2026_08_22_POLYDIM_47_NOCHE"

wb_path = os.path.join(dir_v47, "WHITEBOOK_POLYDIM_V47.md")
cpp_path = os.path.join(dir_v47, "slerp_kernel_v47.cpp")
rs_path = os.path.join(dir_v47, "lib_v47.rs")
build_py_path = os.path.join(dir_v47, "build.py")
cargo_toml_path = os.path.join(dir_v47, "Cargo.toml")
motor_path = os.path.join(dir_v47, "polydim_motor_v47.py")
suite_path = os.path.join(dir_v47, "polydim_suite_v47.py")
monolith_path = os.path.join(dir_v47, "polydim_v47_monolito.py")

with open(wb_path, "r", encoding="utf-8") as f: wb_txt = f.read()
with open(cpp_path, "r", encoding="utf-8") as f: cpp_txt = f.read()
with open(rs_path, "r", encoding="utf-8") as f: rs_txt = f.read()
with open(build_py_path, "r", encoding="utf-8") as f: build_py_txt = f.read()
with open(cargo_toml_path, "r", encoding="utf-8") as f: cargo_toml_txt = f.read()
with open(motor_path, "r", encoding="utf-8") as f: motor_txt = f.read()
with open(suite_path, "r", encoding="utf-8") as f: suite_txt = f.read()

parts = [
    "# polydim_v47_monolito.py\n# MONOLITO UNIFICADO POLYDIM V47.0 - COMPUTABILIDAD GEOMETRICA NATIVA EN S^(D-1)\n# " + "="*76 + "\n\n",
    '"""\n' + wb_txt + '\n"""\n\n',
    'BUILD_PY_SCRIPT = r"""\n' + build_py_txt + '\n"""\n\n',
    'CARGO_TOML_SPEC = r"""\n' + cargo_toml_txt + '\n"""\n\n',
    'SLERP_KERNEL_CPP_SOURCE = r"""\n' + cpp_txt + '\n"""\n\n',
    'LIB_V47_RUST_SOURCE = r"""\n' + rs_txt + '\n"""\n\n',
    motor_txt + "\n\n",
    suite_txt + "\n\n",
    'if __name__ == "__main__":\n    run_suite()\n'
]

full_monolith = "".join(parts)

with open(monolith_path, "w", encoding="utf-8") as f:
    f.write(full_monolith)

print(f"Monolith V47 generated successfully: {os.path.getsize(monolith_path)} bytes")
