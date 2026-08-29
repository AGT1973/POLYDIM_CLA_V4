# PROTOCOLO DE PRUEBAS POR SERVIDOR Y SILICIO (POLYDIM V78)

**Entrega:** POLYDIM V78 APOCALYPSE  
**Directorio Autorizado:** `E:\POLYDIM_EINSOF\ENTREGA_V78_APOCALYPSE_\`  
**Fecha:** 28 de Agosto de 2026  

---

## 📌 1. REGLA ZERO-WASTE DE EJECUCIÓN LOCAL (REGLA 11)

Está terminantemente prohibido compilar o ejecutar scripts de POLYDIM desde unidades sincronizadas por la nube como Google Drive (`I:\`). Todas las ejecuciones deben realizarse en discos físicos locales de alta velocidad (`E:\` o `D:\`).

---

## 💻 2. PRUEBAS EN SERVIDOR LOCAL (CPU WINDOWS / LINUX)

### 2.1 Compilación Manual del Kernel C++ (MSVC / GCC)
```powershell
# Windows MSVC
cl /O2 /std:c++20 /EHsc /LD kernel_cpp_v78.cpp /Fe:polydim_kernel_cpp_v78.dll

# Linux GCC / Clang
g++ -O3 -std=c++20 -fPIC -shared kernel_cpp_v78.cpp -o polydim_kernel_cpp_v78.so
```

### 2.2 Compilación Manual del Kernel Rust
```powershell
# Windows MSVC Target
rustc.exe --crate-type cdylib -C opt-level=3 kernel_rust_v78.rs -o polydim_kernel_rust_v78.dll

# Linux Target
rustc --crate-type cdylib -C opt-level=3 kernel_rust_v78.rs -o polydim_kernel_rust_v78.so
```

### 2.3 Ejecución del Autodiagnóstico del Monolito
```powershell
python polydim_v78_monolito.py
```

---

## 🚀 3. PRUEBAS EN GOOGLE COLAB / KAGGLE (GPU T4 / A100)

### 3.1 Verificación de Entorno FP64 y JAX GPU
```python
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

print("Devices disponibles:", jax.devices())
```

### 3.2 Benchmark de Escalamiento Cayley-SMW ($D = 10^6$)
```python
from polydim_v78_monolito import CliffordRotors, GeodesicKernels

D = 1_000_000
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (D,), dtype=jnp.float64)
x = x / jnp.linalg.norm(x)

U = jax.random.normal(key, (D,), dtype=jnp.float64)
V = jax.random.normal(key, (D,), dtype=jnp.float64)

# Medición de tiempo de compilación y estado estable
%timeit -n 10 -r 3 CliffordRotors.apply_spherical_rotor(x, U, V, alpha=0.01).block_until_ready()
```

---

## 📊 4. MÉTRICAS DE ACEPTACIÓN SOTA V78

| Prueba | Métrica de Éxito | Condición de Fallo |
| :--- | :--- | :--- |
| Reversibilidad $S^{D-1}$ ($D=10^5$) | $\|y - \text{Exp}_x(\text{Log}_x(y))\| < 10^{-6}$ | Error $\ge 10^{-5}$ o NaNs |
| Conservación de Norma Rotor | $\| \text{Rotor}(x) \| = 1.0 \pm 10^{-8}$ | $\| \text{Rotor}(x) \| \neq 1.0$ |
| Detección PMTP Anti-DoS | Rechazo de payloads $> 100$ MB | OOM o lectura de buffer gigantesco |
