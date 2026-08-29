@echo off
echo Compilando Kernels V79...
rustc --crate-type cdylib -C opt-level=3 kernel_rust_v79.rs -o polydim_kernel_rust_v79.dll
echo Finalizado.
