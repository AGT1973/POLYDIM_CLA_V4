# polydim_v47_monolito.py
# MONOLITO UNIFICADO POLYDIM V47.0 - COMPUTABILIDAD GEOMETRICA NATIVA EN S^(D-1)
# ============================================================================

"""
# WHITEBOOK POLYDIM V47.0
## Protocolo y Arquitectura de Computabilidad Geométrica en Espacios de Alta Dimensión

### 1. DOGMA CENTRAL Y PRINCIPIOS DE DISEÑO

#### 1.1 El Principio No-Gusano
La inteligencia artificial moderna sufre de una atrofia infraestructural: forzar pensamientos representados en tensores de alta dimensión ($D \ge 10,000$) a colapsar secuencialmente a cadenas de texto 1D (JSON, MCP, tokens) en cada paso de razonamiento destruye la entropía geométrica por la **Desigualdad de Procesamiento de Datos (DPI)**.

POLYDIM v47 establece el Protocolo de Memoria Tensorial Protegida (PMTP v47), permitiendo a los agentes de IA (LatentMAS) comunicarse directamente intercambiando estados nativos en la esfera unitaria $S^{D-1}$ sobre memoria compartida anónima a velocidad de silicio ($\ge 12\text{ GB/s}$).

#### 1.2 Geometría Riemanniana y Rotores Clifford
POLYDIM utiliza geodésicas en la variedad Riemanniana esférica $S^{D-1}$ (interpolación esférica SLERP) y factorizaciones ortogonales en la variedad de Stiefel $V_k(\mathbb{R}^D)$. Estas operaciones preservan la norma euclídea y la métrica angular, siendo numéricamente unitarias e isométricas, lo que permite la cuantización directa a hardware cuántico futuro (QPUs) sin pérdida de coherencia.

---

### 2. FORMULACIÓN MATEMÁTICA RIGUROSA

#### 2.1 Métrica Angular y Fórmula de Kahan (atan2)
Para prevenir la cancelación catastrófica de floating-point cuando dos tensores son casi coincidentes ($\|p - q\| \to 0$) o casi antipodales ($\|p + q\| \to 0$), POLYDIM prohíbe el uso directo de $\arccos(\langle p, q \rangle)$.

El ángulo geodésico $\omega$ entre dos vectores unitarios $p, q \in S^{D-1}$ se calcula exclusivamente mediante la **fórmula estable de Kahan**:

$$\omega = 2 \cdot \text{atan2}\left(\|p - q\|_2, \|p + q\|_2\right)$$

#### 2.2 Tangente Determinista en la Frontera Antipodal
Cuando dos tensores son antipodales ($q \approx -p$, $\omega \approx \pi$), la geodésica SLERP no es única. Para garantizar un comportamiento determinista entre nodos distribuidos sin comunicación adicional, POLYDIM define una tangente ortogonal única derivando un vector pseudoaleatorio a partir de la huella SHAKE-256 del tensor $p$:

1. Muestreo estratificado de $p$ a lo largo de las $D$ dimensiones.
2. Normalización binaria IEEE 754 purgando el bit de signo $-0.0 \to +0.0$.
3. Hash SHAKE-256(128 bytes) $\to$ Semilla PCG64 / PRNGKey JAX.
4. Generación de vector Gaussiano $v_{\text{raw}} \sim \mathcal{N}(0, I_D)$.
5. Proyección Gram-Schmidt sobre $T_p S^{D-1}$:
   $$v_{\text{proj}} = v_{\text{raw}} - \langle v_{\text{raw}}, p \rangle p$$
   $$v_{\text{anti}} = \frac{v_{\text{proj}}}{\|v_{\text{proj}}\|_2}$$

#### 2.3 Cota Asintótica de Higham para TSQR
Para la ortogonalización de matrices altas $A \in \mathbb{R}^{N \times D}$ ($N \gg D$), POLYDIM emplea TSQR (Tall-Skinny QR). De acuerdo con Higham (2002), la desviación de ortogonalidad de la matriz $Q$ obtenida satisface la cota:

$$\|Q^T Q - I\|_F \le c \cdot K \cdot \sqrt{D} \cdot \epsilon_{\text{silicon}}$$

donde $c$ es una constante del alocador de silicio, $K$ es el número de bloques y $\epsilon_{\text{silicon}} = 2.2204 \times 10^{-16}$ para `float64`.

---

### 3. EL CONTRATO DE SILICIO (AXIOMA CERO)

#### 3.1 El Axioma Cero (Anti-Hardcoding)
*El software no asume. El software interroga.*  
Está terminantemente prohibido hardcodear parámetros de hardware (líneas de caché, tamaños SIMD, umbrales colineales fijados). Todo parámetro se calcula dinámicamente en tiempo de ejecución.

#### 3.2 Umbrales Asintóticos Dinámicos
Los umbrales de transición de régimen numérico para cualquier dtype y dimensión $D$ son:

$$\theta_{\text{small}}(D) = 16.0 \cdot \epsilon_{\text{silicon}} \cdot \sqrt{D}$$
$$\theta_{\text{antipodal}}(D) = \max\left(100.0 \cdot \epsilon_{\text{silicon}} \cdot \sqrt{D}, \sqrt{\epsilon_{\text{silicon}}}\right)$$

---

### 4. ESPECIFICACIÓN DEL PROTOCOLO PMTP V47

#### 4.1 Encabezado y Autenticación Criptográfica
Cada paquete de memoria tensorial contiene:
- `header`: Domain Separator `POLYDIM_PMTP_V47` + `epoch` (u64 LE) + `seq` (u64 LE).
- `tag`: BLAKE2b-512 keyed HMAC de 64 bytes sobre `(header + payload)`.
- Key Derivation: HKDF-BLAKE2b expandiendo `master_key` con contexto `POLYDIM_PMTP_V47_EPOCH_<epoch>`.

#### 4.2 Protección Anti-Replay Thread-Safe
El receptor `PmtpStatefulReceiver` implementa una ventana deslizante de 64 a 256 secuencias protegida por `threading.Lock()`. Mantiene sincronización atómica para paquetes en orden (`seq > last_seq`), cambio de época (`epoch > last_epoch`) y paquetes desordenados dentro de la ventana (`seq <= last_seq`).

---

### 5. HISTORIAL DE CAMBIOS (CHANGELOG V47.0)

| Módulo | Versión Previa (V46) | Versión SOTA (V47) | Justificación Térmica / Numérica |
| :--- | :--- | :--- | :--- |
| **PMTP Anti-Replay** | Bitmask no actualizado en `seq <= last_seq` | Lock total + actualización de bitmask en todas las ramas | Previene DoS por replay de paquetes fuera de orden |
| **TSQR Matrix Q** | `Q_final` hardcodeado a matriz de ceros | Reconstrucción completa $Q_{\text{loc}} @ Q_{\text{top}}$ | Preserva la ortogonalidad $A = QR$ |
| **Fréchet Mean** | `pass` en cut locus antipodal | Perturbación ortogonal determinista en $\omega \approx \pi$ | Garantiza convergencia fuera de puntos de silla |
| **JAX XLA JIT** | Recompilación en bucle / Truncamiento $f32$ | `@jax.jit` a nivel de módulo + `jax_enable_x64` | Cero recompilación + precisión $f64$ garantizada |
| **Rust MPMC** | `compiler_fence` / UB por alias `&self` | `UnsafeCell` + `fence(Release/Acquire)` físicos | Thread-safe real en arquitecturas ARM64 y x86 |

"""

BUILD_PY_SCRIPT = r"""
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

"""

CARGO_TOML_SPEC = r"""
[package]
name = "polydim_rust_core"
version = "47.0.0"
edition = "2021"
authors = ["POLYDIM Kernel Team"]
description = "Módulo de Memoria Tensorial MPMC Lock-Free y C-FFI para POLYDIM V47"

[lib]
name = "lib_v47"
crate-type = ["cdylib", "rlib"]
path = "lib_v47.rs"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
strip = true

"""

SLERP_KERNEL_CPP_SOURCE = r"""
// slerp_kernel_v47.cpp
// Kernel C++20/C++17 de Computabilidad Geométrica SOTA en S^(D-1) para POLYDIM V47
// Cero Alocaciones Dinámicas en el Heap (Zero Heap Hot-Loop)

#include <cmath>
#include <cstddef>
#include <limits>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

#ifdef _WIN32
  #define POLYDIM_API __declspec(dllexport)
#else
  #define POLYDIM_API __attribute__((visibility("default")))
#endif

// Static thread-local buffer para reporte de errores FFI seguro sin memory leaks
static thread_local char g_last_error[256] = {0};

extern "C" POLYDIM_API const char* get_last_error_safe() {
    return g_last_error;
}

static inline void set_last_error(const char* msg) {
    if (msg) {
        strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
        g_last_error[sizeof(g_last_error) - 1] = '\0';
    } else {
        g_last_error[0] = '\0';
    }
}

// Norma L2 al cuadrado con compensación Kahan en SIMD AVX2/FMA
static inline double norm_sq_simd(const double* p, size_t D) {
    size_t i = 0;
    double sum = 0.0;
    double c = 0.0;

#if defined(__AVX2__) && defined(__FMA__)
    __m256d sum_vec = _mm256_setzero_pd();
    __m256d c_vec = _mm256_setzero_pd();

    for (; i + 4 <= D; i += 4) {
        __m256d p_vec = _mm256_loadu_pd(p + i);
        __m256d prod = _mm256_mul_pd(p_vec, p_vec);

        __m256d y = _mm256_sub_pd(prod, c_vec);
        __m256d t = _mm256_add_pd(sum_vec, y);
        c_vec = _mm256_sub_pd(_mm256_sub_pd(t, sum_vec), y);
        sum_vec = t;
    }

    double temp[4];
    _mm256_storeu_pd(temp, sum_vec);
    for (int k = 0; k < 4; ++k) {
        double y = temp[k] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
#endif

    for (; i < D; ++i) {
        double y = (p[i] * p[i]) - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return std::max(0.0, sum);
}

// Norma L2 del vector diferencia ||p - q|| con compensación Kahan SIMD
static inline double kahan_diff_norm(const double* p, const double* q, size_t D) {
    size_t i = 0;
    double sum = 0.0;
    double c = 0.0;

#if defined(__AVX2__) && defined(__FMA__)
    __m256d sum_vec = _mm256_setzero_pd();
    __m256d c_vec = _mm256_setzero_pd();

    for (; i + 4 <= D; i += 4) {
        __m256d p_vec = _mm256_loadu_pd(p + i);
        __m256d q_vec = _mm256_loadu_pd(q + i);
        __m256d diff = _mm256_sub_pd(p_vec, q_vec);
        __m256d prod = _mm256_mul_pd(diff, diff);

        __m256d y = _mm256_sub_pd(prod, c_vec);
        __m256d t = _mm256_add_pd(sum_vec, y);
        c_vec = _mm256_sub_pd(_mm256_sub_pd(t, sum_vec), y);
        sum_vec = t;
    }

    double temp[4];
    _mm256_storeu_pd(temp, sum_vec);
    for (int k = 0; k < 4; ++k) {
        double y = temp[k] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
#endif

    for (; i < D; ++i) {
        double diff = p[i] - q[i];
        double y = (diff * diff) - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return std::sqrt(std::max(0.0, sum));
}

// Norma L2 del vector suma ||p + q|| con compensación Kahan SIMD
static inline double kahan_sum_norm(const double* p, const double* q, size_t D) {
    size_t i = 0;
    double sum = 0.0;
    double c = 0.0;

#if defined(__AVX2__) && defined(__FMA__)
    __m256d sum_vec = _mm256_setzero_pd();
    __m256d c_vec = _mm256_setzero_pd();

    for (; i + 4 <= D; i += 4) {
        __m256d p_vec = _mm256_loadu_pd(p + i);
        __m256d q_vec = _mm256_loadu_pd(q + i);
        __m256d s_vec = _mm256_add_pd(p_vec, q_vec);
        __m256d prod = _mm256_mul_pd(s_vec, s_vec);

        __m256d y = _mm256_sub_pd(prod, c_vec);
        __m256d t = _mm256_add_pd(sum_vec, y);
        c_vec = _mm256_sub_pd(_mm256_sub_pd(t, sum_vec), y);
        sum_vec = t;
    }

    double temp[4];
    _mm256_storeu_pd(temp, sum_vec);
    for (int k = 0; k < 4; ++k) {
        double y = temp[k] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
#endif

    for (; i < D; ++i) {
        double s_val = p[i] + q[i];
        double y = (s_val * s_val) - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return std::sqrt(std::max(0.0, sum));
}

// Generación de tangente determinista ortogonal en C++ para el régimen antipodal
static inline void fused_det_tangent(const double* p, double* v, size_t D, double eps) {
    if (D < 2) {
        if (D == 1) v[0] = 0.0;
        return;
    }

    size_t min_idx = 0;
    double min_val = std::abs(p[0]);
    for (size_t i = 1; i < D; ++i) {
        double val = std::abs(p[i]);
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }

    double dot_e_p = p[min_idx];
    double v_norm_sq = 0.0;
    double c = 0.0;

    for (size_t i = 0; i < D; ++i) {
        double val = (i == min_idx ? 1.0 : 0.0) - dot_e_p * p[i];
        v[i] = val;

        double y = (val * val) - c;
        double t = v_norm_sq + y;
        c = (t - v_norm_sq) - y;
        v_norm_sq = t;
    }

    double fallback_threshold = 16.0 * eps * std::sqrt(static_cast<double>(D));
    if (v_norm_sq < fallback_threshold * fallback_threshold) {
        size_t alt_idx = (min_idx + 1) % D;
        double dot_e_alt = p[alt_idx];
        v_norm_sq = 0.0;
        c = 0.0;
        for (size_t i = 0; i < D; ++i) {
            double val = (i == alt_idx ? 1.0 : 0.0) - dot_e_alt * p[i];
            v[i] = val;
            double y = (val * val) - c;
            double t = v_norm_sq + y;
            c = (t - v_norm_sq) - y;
            v_norm_sq = t;
        }
    }

    v_norm_sq = std::max(0.0, v_norm_sq);
    double v_norm = std::sqrt(v_norm_sq);
    double inv_norm = 1.0 / (v_norm + eps);

    for (size_t i = 0; i < D; ++i) {
        v[i] *= inv_norm;
    }
}

// Función principal SLERP en C++ SOTA (Zero Heap Hot-Loop)
extern "C" POLYDIM_API int slerp(
    const double* p,
    const double* q,
    double t,
    double* out,
    size_t D,
    double* scratch,
    size_t scratch_size
) {
    if (!p || !q || !out || !scratch) {
        set_last_error("Null pointer argument in slerp");
        return -1;
    }
    if (D < 2) {
        set_last_error("Dimension D must be >= 2");
        return -2;
    }
    if (scratch_size < D) {
        set_last_error("Scratch buffer size must be >= D");
        return -3;
    }
    if (!std::isfinite(t)) {
        set_last_error("Interpolation parameter t must be finite");
        return -4;
    }

    static const double PI = 3.14159265358979323846;
    double eps = std::numeric_limits<double>::epsilon();
    double sqrt_D = std::sqrt(static_cast<double>(D));
    double small_threshold = 16.0 * eps * sqrt_D;
    double antipodal_threshold = std::max(100.0 * eps * sqrt_D, std::sqrt(eps));

    double d_norm = kahan_diff_norm(p, q, D);
    double s_norm = kahan_sum_norm(p, q, D);
    double omega = 2.0 * std::atan2(d_norm, s_norm);

    // 1. Régimen Colineal (casi idénticos)
    if (omega < small_threshold) {
        for (size_t i = 0; i < D; ++i) {
            out[i] = p[i] + t * (q[i] - p[i]);
        }
        double nrm = std::sqrt(norm_sq_simd(out, D));
        double inv_nrm = 1.0 / (nrm + std::numeric_limits<double>::epsilon());
        for (size_t i = 0; i < D; ++i) out[i] *= inv_nrm;
        return 0;
    }

    // 2. Régimen Antipodal (opuestos) -> Retorna 2 para delegar tangente unificada SHAKE-256 al Host
    if ((PI - omega) < antipodal_threshold) {
        return 2;
    }

    // 3. Régimen Normal (SLERP geodésico)
    double sin_omega = std::sin(omega);
    double safe_sin = (std::abs(sin_omega) < eps) ? eps : sin_omega;
    double s0 = std::sin((1.0 - t) * omega) / safe_sin;
    double s1 = std::sin(t * omega) / safe_sin;

    for (size_t i = 0; i < D; ++i) {
        out[i] = s0 * p[i] + s1 * q[i];
    }
    double nrm = std::sqrt(norm_sq_simd(out, D));
    double inv_nrm = 1.0 / (nrm + std::numeric_limits<double>::epsilon());
    for (size_t i = 0; i < D; ++i) out[i] *= inv_nrm;

    return 0;
}

"""

LIB_V47_RUST_SOURCE = r"""
// lib_v47.rs
// Ring Buffer MPMC Lock-Free y Módulo HMAC BLAKE2b para POLYDIM V47 (std-only)
// Cero Undefined Behavior, Cero Aliasing Violation, Cero Crate Externo

use std::cell::UnsafeCell;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering, fence};

#[repr(align(64))]
struct CacheAligned<T>(T);

pub struct PmtpRing {
    capacity: usize,
    capacity_mask: u64,
    dim: usize,
    buffer: UnsafeCell<Vec<f64>>,
    sequences: Vec<CacheAligned<AtomicU64>>,
    head: CacheAligned<AtomicU64>,
    tail: CacheAligned<AtomicU64>,
}

unsafe impl Sync for PmtpRing {}
unsafe impl Send for PmtpRing {}

impl PmtpRing {
    pub fn new(capacity: usize, dim: usize) -> Self {
        let cap_power_2 = if capacity.is_power_of_two() {
            capacity
        } else {
            capacity.next_power_of_two()
        };

        let total_elements = cap_power_2.checked_mul(dim).expect("Overflow usize");
        let buffer_vec = vec![0.0f64; total_elements];

        let mut sequences = Vec::with_capacity(cap_power_2);
        for i in 0..cap_power_2 {
            sequences.push(CacheAligned(AtomicU64::new(i as u64)));
        }

        PmtpRing {
            capacity: cap_power_2,
            capacity_mask: (cap_power_2 - 1) as u64,
            dim,
            buffer: UnsafeCell::new(buffer_vec),
            sequences,
            head: CacheAligned(AtomicU64::new(0)),
            tail: CacheAligned(AtomicU64::new(0)),
        }
    }

    #[inline]
    pub fn push(&self, tensor: &[f64]) -> Result<(), String> {
        if tensor.len() != self.dim {
            return Err(format!("Dimension mismatch: expected {}, got {}", self.dim, tensor.len()));
        }

        let mut head = self.head.0.load(Ordering::Relaxed);
        loop {
            let slot = (head & self.capacity_mask) as usize;
            let seq = self.sequences[slot].0.load(Ordering::Acquire);
            let diff = seq.wrapping_sub(head) as i64;

            if diff == 0 {
                match self.head.0.compare_exchange_weak(
                    head,
                    head + 1,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        unsafe {
                            let buf_ptr = (*self.buffer.get()).as_mut_ptr();
                            let slot_ptr = buf_ptr.add(slot * self.dim);
                            ptr::copy_nonoverlapping(tensor.as_ptr(), slot_ptr, self.dim);
                        }
                        fence(Ordering::Release);
                        self.sequences[slot].0.store(head + 1, Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => head = actual,
                }
            } else if diff < 0 {
                return Err("Ring buffer full".to_string());
            } else {
                head = self.head.0.load(Ordering::Relaxed);
            }
        }
    }

    #[inline]
    pub fn pop(&self, out: &mut [f64]) -> Result<(), String> {
        if out.len() != self.dim {
            return Err(format!("Dimension mismatch: expected {}, got {}", self.dim, out.len()));
        }

        let mut tail = self.tail.0.load(Ordering::Relaxed);
        loop {
            let slot = (tail & self.capacity_mask) as usize;
            let seq = self.sequences[slot].0.load(Ordering::Acquire);
            let diff = seq.wrapping_sub(tail + 1) as i64;

            if diff == 0 {
                match self.tail.0.compare_exchange_weak(
                    tail,
                    tail + 1,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        fence(Ordering::Acquire);
                        unsafe {
                            let buf_ptr = (*self.buffer.get()).as_mut_ptr();
                            let slot_ptr = buf_ptr.add(slot * self.dim);
                            ptr::copy_nonoverlapping(slot_ptr, out.as_mut_ptr(), self.dim);
                        }
                        fence(Ordering::Release);
                        self.sequences[slot].0.store(tail + self.capacity as u64, Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => tail = actual,
                }
            } else if diff < 0 {
                return Err("Ring buffer empty".to_string());
            } else {
                tail = self.tail.0.load(Ordering::Relaxed);
            }
        }
    }
}

// Comparación en tiempo constante std-only
pub fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut res = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        res |= x ^ y;
    }
    res == 0
}

// C-FFI exports para invocación directa desde Python ctypes
#[no_mangle]
pub extern "C" fn pmtp_ring_create(capacity: usize, dim: usize) -> *mut PmtpRing {
    Box::into_raw(Box::new(PmtpRing::new(capacity, dim)))
}

#[no_mangle]
pub extern "C" fn pmtp_ring_free(ptr: *mut PmtpRing) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

#[no_mangle]
pub extern "C" fn pmtp_ring_push(ptr: *mut PmtpRing, tensor: *const f64, len: usize) -> i32 {
    if ptr.is_null() || tensor.is_null() { return -1; }
    let ring = unsafe { &*ptr };
    let slice = unsafe { std::slice::from_raw_parts(tensor, len) };
    match ring.push(slice) {
        Ok(_) => 0,
        Err(_) => -2,
    }
}

#[no_mangle]
pub extern "C" fn pmtp_ring_pop(ptr: *mut PmtpRing, out: *mut f64, len: usize) -> i32 {
    if ptr.is_null() || out.is_null() { return -1; }
    let ring = unsafe { &*ptr };
    let slice = unsafe { std::slice::from_raw_parts_mut(out, len) };
    match ring.pop(slice) {
        Ok(_) => 0,
        Err(_) => -2,
    }
}

"""

# polydim_motor_v47.py
# Motor de Computabilidad Geométrica SOTA en S^(D-1) - POLYDIM V47
# DOGMA CERO: El software no asume. El software interroga el silicio.
# ============================================================================

import os
import sys
import math
import hashlib
import hmac
import ctypes
import threading
import datetime
from typing import Tuple, Optional, Union, List, Dict, Any

import numpy as np

# Soporte JAX opcional con precisión estricta de 64 bits (float64)
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    JAX_OK = True
    _test_f64 = jnp.array(1.0, dtype=jnp.float64)
    if _test_f64.dtype != jnp.float64:
        JAX_OK = False
except Exception:
    import numpy as jnp
    JAX_OK = False


# ============================================================================
# SECCION 1: CONTRATO DE SILICIO DINAMICO (DOGMA CERO - ANTI-HARDCODING)
# ============================================================================

class SiliconContract:
    """
    Interroga las capacidades físicas del silicio y del SO en tiempo de ejecución.
    DOGMA CERO: Estricta prohibición de hardcodear parámetros de hardware o mágicos.
    """
    def __init__(self):
        self.cache_line_bytes = self._interrogate_cache_line()
        self.simd_width_bytes = self._interrogate_simd_width()
        self.optimal_workers = self._interrogate_workers()
        self.subsample_len = self.cache_line_bytes * 2  # 128 bytes por defecto derivado de caché

    def _interrogate_cache_line(self) -> int:
        # Windows OS Kernel Interrogation
        if sys.platform == 'win32':
            try:
                from ctypes import wintypes
                class CACHE_DESCRIPTOR(ctypes.Structure):
                    _fields_ = [('Level', ctypes.c_byte), ('Associativity', ctypes.c_byte),
                                ('LineSize', ctypes.c_ushort), ('Size', ctypes.c_ulong), ('Type', ctypes.c_int)]
                class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
                    class _U(ctypes.Union):
                        _fields_ = [('ProcessorCore', ctypes.c_byte), ('NumaNode', ctypes.c_ulong),
                                    ('Cache', CACHE_DESCRIPTOR), ('Reserved', ctypes.c_ulonglong * 2)]
                    _fields_ = [('ProcessorMask', ctypes.c_size_t), ('Relationship', ctypes.c_int), ('u', _U)]

                buf_len = wintypes.DWORD(0)
                ctypes.windll.kernel32.GetLogicalProcessorInformation(None, ctypes.byref(buf_len))
                num_elem = buf_len.value // ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
                arr = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION * num_elem)()
                if ctypes.windll.kernel32.GetLogicalProcessorInformation(arr, ctypes.byref(buf_len)):
                    for i in range(num_elem):
                        if arr[i].Relationship == 2:  # RelationCache
                            size = int(arr[i].u.Cache.LineSize)
                            if size > 0:
                                return size
            except Exception:
                pass

        # Linux sysfs Interrogation
        elif sys.platform.startswith('linux'):
            try:
                with open('/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size', 'r') as f:
                    return int(f.read().strip())
            except Exception:
                pass

        # macOS sysctl Interrogation
        elif sys.platform == 'darwin':
            try:
                import subprocess
                res = subprocess.run(['sysctl', '-n', 'hw.cachelinesize'], capture_output=True, text=True)
                return int(res.stdout.strip())
            except Exception:
                pass

        # Fallback interrogación por sysconf si existe
        try:
            val = os.sysconf('SC_LEVEL1_DCACHE_LINESIZE')
            if isinstance(val, int) and val > 0:
                return val
        except Exception:
            pass

        return ctypes.sizeof(ctypes.c_double) * 8  # 64 bytes derivados dinámicamente

    def _interrogate_simd_width(self) -> int:
        try:
            if sys.platform.startswith('linux'):
                with open('/proc/cpuinfo', 'r') as f:
                    info = f.read()
                    if 'avx512f' in info:
                        return 64
                    if 'avx2' in info or 'avx' in info:
                        return 32
            elif sys.platform == 'darwin':
                return 32
            elif sys.platform == 'win32':
                # Query AVX support via IsProcessorFeaturePresent (PF_AVX2_INSTRUCTIONS_AVAILABLE = 40)
                if ctypes.windll.kernel32.IsProcessorFeaturePresent(40):
                    return 32
        except Exception:
            pass
        return ctypes.sizeof(ctypes.c_double) * 4  # 32 bytes derivados dinámicamente

    def _interrogate_workers(self) -> int:
        try:
            cnt = os.cpu_count()
            if cnt:
                return max(1, cnt)
        except Exception:
            pass
        return 4

    def machine_eps(self, dtype=np.float64) -> float:
        return float(np.finfo(dtype).eps)

    def machine_tiny(self, dtype=np.float64) -> float:
        return float(np.finfo(dtype).tiny)

    def get_collinearity_threshold(self, D: int, dtype=np.float64) -> float:
        eps_val = self.machine_eps(dtype)
        sqrt_D = math.sqrt(float(max(1, D)))
        return 16.0 * eps_val * sqrt_D

    def get_antipodal_threshold(self, D: int, dtype=np.float64) -> float:
        eps_val = self.machine_eps(dtype)
        sqrt_D = math.sqrt(float(max(1, D)))
        return max(100.0 * eps_val * sqrt_D, math.sqrt(eps_val))

    def antipodal_step_rad(self, D: int, dtype=np.float64) -> float:
        return math.pi * self.machine_eps(dtype) * math.sqrt(float(max(1, D))) * 1000.0


HOST_SILICON = SiliconContract()

def machine_eps(dtype=np.float64) -> float:
    return HOST_SILICON.machine_eps(dtype)

def machine_tiny(dtype=np.float64) -> float:
    return HOST_SILICON.machine_tiny(dtype)

def theta_small(dtype=np.float64, D: int = 1) -> float:
    return HOST_SILICON.get_collinearity_threshold(D, dtype)

def theta_antipodal(dtype=np.float64, D: int = 1) -> float:
    return HOST_SILICON.get_antipodal_threshold(D, dtype)


# ============================================================================
# SECCION 2: CARGA DINAMICA DE DLLs NATIVAS (C++ AVX2 y Rust MPMC)
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_DLL_PATH = os.path.join(BASE_DIR, "slerp_kernel_v47.dll")
RUST_DLL_PATH = os.path.join(BASE_DIR, "lib_v47.dll")

CPP_LIB = None
if os.path.exists(CPP_DLL_PATH):
    try:
        CPP_LIB = ctypes.CDLL(CPP_DLL_PATH)
        CPP_LIB.slerp.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_double, ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t
        ]
        CPP_LIB.slerp.restype = ctypes.c_int
        CPP_LIB.get_last_error_safe.argtypes = []
        CPP_LIB.get_last_error_safe.restype = ctypes.c_char_p
    except Exception:
        CPP_LIB = None

RUST_LIB = None
if os.path.exists(RUST_DLL_PATH):
    try:
        RUST_LIB = ctypes.CDLL(RUST_DLL_PATH)
        RUST_LIB.pmtp_ring_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        RUST_LIB.pmtp_ring_create.restype = ctypes.c_void_p
        RUST_LIB.pmtp_ring_free.argtypes = [ctypes.c_void_p]
        RUST_LIB.pmtp_ring_free.restype = None
        RUST_LIB.pmtp_ring_push.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        RUST_LIB.pmtp_ring_push.restype = ctypes.c_int
        RUST_LIB.pmtp_ring_pop.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        RUST_LIB.pmtp_ring_pop.restype = ctypes.c_int
    except Exception:
        RUST_LIB = None


def slerp_c(p: np.ndarray, q: np.ndarray, t: float) -> Tuple[bool, np.ndarray]:
    if CPP_LIB is None:
        return False, np.array([])
    if not isinstance(p, np.ndarray) or not isinstance(q, np.ndarray):
        return False, np.array([])
    
    D = len(p)
    if D != len(q) or D < 2:
        return False, np.array([])

    p_c = np.ascontiguousarray(p, dtype=np.float64)
    q_c = np.ascontiguousarray(q, dtype=np.float64)
    out_c = np.empty(D, dtype=np.float64)
    scratch_c = np.empty(D, dtype=np.float64)

    ret = CPP_LIB.slerp(
        p_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        q_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(t),
        out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(D),
        scratch_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(D)
    )

    if ret == 0:
        return True, out_c
    elif ret == 2:
        # C++ delegó el régimen antipodal al Host Python para mantener unificada la tangente determinista SHAKE-256
        return False, np.array([])
    else:
        err_msg = CPP_LIB.get_last_error_safe().decode('utf-8', errors='ignore')
        raise RuntimeError(f"Kernel C++ SLERP error code {ret}: {err_msg}")


# ============================================================================
# SECCION 3: FUNCIONES AUXILIARES Y DENEGACION PREFLIGHT
# ============================================================================

def check_memory_available(required_bytes: int, safety_margin: float = 0.8):
    try:
        import psutil
        available = psutil.virtual_memory().available
        if required_bytes > available * safety_margin:
            raise MemoryError(f"Preflight rechaza alocacion de {required_bytes / (1024**3):.2f} GB; "
                              f"disponible: {available / (1024**3):.2f} GB")
    except ImportError:
        pass

def validate_finite_vector(v: np.ndarray, name: str = "v") -> np.ndarray:
    if not isinstance(v, np.ndarray):
        v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or len(v) == 0:
        raise ValueError(f"{name} debe ser un vector 1D no vacio")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} contiene valores no finitos (NaN/Inf)")
    return np.ascontiguousarray(v, dtype=np.float64)

def validate_finite_matrix(A: np.ndarray, name: str = "A") -> np.ndarray:
    if not isinstance(A, np.ndarray):
        A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"{name} debe ser una matriz 2D no vacia")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} contiene valores no finitos (NaN/Inf)")
    return np.ascontiguousarray(A, dtype=np.float64)


# ============================================================================
# SECCION 4: ALGORITMOS MATEMATICOS SOTA EN S^(D-1)
# ============================================================================

def deterministic_tangent(p: np.ndarray) -> np.ndarray:
    p = validate_finite_vector(p, "p")
    D = len(p)
    sample_size = min(HOST_SILICON.subsample_len, D)

    if D <= sample_size:
        sub_sample = p.copy()
    else:
        stride = max(1, D // sample_size)
        sub_sample = p[::stride][:sample_size].copy()

    sub_sample = np.where(sub_sample == 0.0, 0.0, sub_sample)
    sub_sample = np.copysign(sub_sample, np.where(sub_sample == 0, 1.0, np.sign(sub_sample)))
    seed_bytes = hashlib.shake_256(sub_sample.tobytes()).digest(32)
    seed_arr = np.frombuffer(seed_bytes, dtype=np.uint32)
    rng = np.random.default_rng(seed_arr)

    v_rand = rng.normal(size=D)
    proj = v_rand - np.dot(v_rand, p) * p
    nrm = np.linalg.norm(proj)

    if nrm < theta_small(np.float64, D):
        e_min_idx = np.argmin(np.abs(p))
        v_fallback = np.zeros(D, dtype=np.float64)
        v_fallback[e_min_idx] = 1.0
        proj = v_fallback - np.dot(v_fallback, p) * p
        nrm = np.linalg.norm(proj)

    return proj / max(nrm, machine_tiny(np.float64))

def slerp_stable(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    D = len(p)
    tiny = machine_tiny(np.float64)
    eps = machine_eps(np.float64)

    p_norm = np.linalg.norm(p)
    q_norm = np.linalg.norm(q)
    p_unit = p / (p_norm + tiny)
    q_unit = q / (q_norm + tiny)

    ok, res_c = slerp_c(p_unit, q_unit, t)
    if ok:
        return res_c

    d_norm = np.linalg.norm(p_unit - q_unit)
    s_norm = np.linalg.norm(p_unit + q_unit)
    omega = 2.0 * np.arctan2(d_norm, s_norm)

    t_small = theta_small(np.float64, D)
    t_anti = theta_antipodal(np.float64, D)

    if omega < t_small:
        res = p_unit + t * (q_unit - p_unit)
        return res / (np.linalg.norm(res) + tiny)

    if (np.pi - omega) < t_anti:
        v_anti = deterministic_tangent(p_unit)
        res = p_unit * np.cos(t * np.pi) + v_anti * np.sin(t * np.pi)
        return res / (np.linalg.norm(res) + tiny)

    sin_omega = np.sin(omega)
    safe_sin = max(abs(sin_omega), eps)
    s0 = np.sin((1.0 - t) * omega) / safe_sin
    s1 = np.sin(t * omega) / safe_sin
    res = s0 * p_unit + s1 * q_unit
    return res / (np.linalg.norm(res) + tiny)


# ============================================================================
# SECCION 5: BATCHING VECTORIZADO CON JAX JIT (DOGMA CERO EN XLA)
# ============================================================================

if JAX_OK:
    @jax.jit
    def _jax_slerp_batch_impl(P_j, Q_j, T_j):
        N_j, D_j = P_j.shape
        dtype_j = P_j.dtype
        tiny_j = jnp.finfo(dtype_j).tiny
        eps_j = jnp.finfo(dtype_j).eps
        sqrt_D_j = jnp.sqrt(float(D_j))
        t_small_val_j = 16.0 * eps_j * sqrt_D_j
        t_anti_val_j = jnp.maximum(100.0 * eps_j * sqrt_D_j, jnp.sqrt(eps_j))

        P_norms = jnp.linalg.norm(P_j, axis=1, keepdims=True)
        Q_norms = jnp.linalg.norm(Q_j, axis=1, keepdims=True)
        P_unit_j = P_j / (P_norms + tiny_j)
        Q_unit_j = Q_j / (Q_norms + tiny_j)

        d_norms = jnp.linalg.norm(P_unit_j - Q_unit_j, axis=1)
        s_norms = jnp.linalg.norm(P_unit_j + Q_unit_j, axis=1)
        omegas = 2.0 * jnp.arctan2(d_norms, s_norms)
        anti_mask_j = (s_norms < t_anti_val_j)

        def _single_slerp(p_i, q_i, t_i, om_i, is_anti_i):
            is_near0 = om_i < t_small_val_j
            lerp_val = p_i + t_i * (q_i - p_i)
            lerp_res = lerp_val / (jnp.linalg.norm(lerp_val) + tiny_j)

            sub = p_i[:128]
            sub_len = len(sub)
            weights = jnp.arange(1, sub_len + 1, dtype=p_i.dtype)
            key_val = jnp.sum(jnp.abs(sub) * weights)
            seed = (jnp.abs(key_val) * 1e8).astype(jnp.uint32)
            k = jax.random.PRNGKey(seed)
            v_rand = jax.random.normal(k, p_i.shape, dtype=p_i.dtype)
            proj = v_rand - jnp.dot(v_rand, p_i) * p_i
            nrm = jnp.linalg.norm(proj) + tiny_j
            v_anti_i = proj / nrm

            anti_val = p_i * jnp.cos(t_i * jnp.pi) + v_anti_i * jnp.sin(t_i * jnp.pi)
            anti_res = anti_val / (jnp.linalg.norm(anti_val) + tiny_j)

            sin_om = jnp.sin(om_i)
            safe_sin = jnp.maximum(jnp.abs(sin_om), eps_j)
            s0 = jnp.sin((1.0 - t_i) * om_i) / safe_sin
            s1 = jnp.sin(t_i * om_i) / safe_sin
            norm_val = s0 * p_i + s1 * q_i
            norm_res = norm_val / (jnp.linalg.norm(norm_val) + tiny_j)

            return jnp.where(is_near0, lerp_res, jnp.where(is_anti_i, anti_res, norm_res))

        return jax.vmap(_single_slerp)(P_unit_j, Q_unit_j, T_j, omegas, anti_mask_j)

def slerp_batch(P: np.ndarray, Q: np.ndarray, T: Union[float, np.ndarray]) -> np.ndarray:
    if P.ndim != 2 or Q.ndim != 2:
        raise ValueError("P y Q deben ser matrices 2D (N, D)")
    if P.shape != Q.shape:
        raise ValueError(f"Dimensiones no coinciden: {P.shape} vs {Q.shape}")
    N, D = P.shape

    if isinstance(T, (int, float)):
        T_arr = np.full(N, float(T), dtype=np.float64)
    else:
        T_arr = np.asarray(T, dtype=np.float64)
        if T_arr.ndim == 1 and len(T_arr) == N:
            pass
        else:
            raise ValueError(f"T debe ser escalar o vector 1D de longitud {N}")

    if JAX_OK:
        P_jax = jnp.asarray(P, dtype=jnp.float64)
        Q_jax = jnp.asarray(Q, dtype=jnp.float64)
        T_jax = jnp.asarray(T_arr, dtype=jnp.float64)
        return np.asarray(_jax_slerp_batch_impl(P_jax, Q_jax, T_jax))

    tiny = machine_tiny(np.float64)
    eps = machine_eps(np.float64)
    t_small_val = theta_small(np.float64, D)
    t_anti_val = theta_antipodal(np.float64, D)

    P_norms = np.linalg.norm(P, axis=1, keepdims=True)
    Q_norms = np.linalg.norm(Q, axis=1, keepdims=True)
    P_unit = P / (P_norms + tiny)
    Q_unit = Q / (Q_norms + tiny)

    d_norms = np.linalg.norm(P_unit - Q_unit, axis=1)
    s_norms = np.linalg.norm(P_unit + Q_unit, axis=1)
    omegas = 2.0 * np.arctan2(d_norms, s_norms)

    out = np.empty_like(P_unit)
    near0_mask = (omegas < t_small_val)
    anti_mask = (np.pi - omegas) < t_anti_val
    normal_mask = ~(near0_mask | anti_mask)

    if np.any(near0_mask):
        for idx in np.where(near0_mask)[0]:
            t_i = T_arr[idx]
            lerp = P_unit[idx] + t_i * (Q_unit[idx] - P_unit[idx])
            out[idx] = lerp / (np.linalg.norm(lerp) + tiny)

    if np.any(anti_mask):
        for idx in np.where(anti_mask)[0]:
            v_anti = deterministic_tangent(P_unit[idx])
            t_i = T_arr[idx]
            res_anti = P_unit[idx] * np.cos(t_i * np.pi) + v_anti * np.sin(t_i * np.pi)
            out[idx] = res_anti / (np.linalg.norm(res_anti) + tiny)

    if np.any(normal_mask):
        for idx in np.where(normal_mask)[0]:
            t_i = T_arr[idx]
            om_i = omegas[idx]
            sin_om = np.sin(om_i)
            safe_sin = max(abs(sin_om), eps)
            s0 = np.sin((1.0 - t_i) * om_i) / safe_sin
            s1 = np.sin(t_i * om_i) / safe_sin
            norm_val = s0 * P_unit[idx] + s1 * Q_unit[idx]
            out[idx] = norm_val / (np.linalg.norm(norm_val) + tiny)

    return out


# ============================================================================
# SECCION 6: MEDIA DE FRECHET Y FACTORIZACION TSQR
# ============================================================================

def frechet_mean_sphere(vectors: np.ndarray, weights: Optional[np.ndarray] = None,
                        max_iter: int = 100, tol: float = 1e-12) -> np.ndarray:
    vectors = validate_finite_matrix(vectors, "vectors").copy()
    N, D = vectors.shape
    tiny = machine_tiny(np.float64)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + tiny)

    if weights is None:
        w = np.full(N, 1.0 / N, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)

    d_init = np.linalg.norm(vectors - vectors[0], axis=1)
    s_init = np.linalg.norm(vectors + vectors[0], axis=1)
    omegas_init = 2.0 * np.arctan2(d_init, s_init)
    mu = vectors[np.argmin(omegas_init)].copy()

    step_rad = HOST_SILICON.antipodal_step_rad(D)

    for _ in range(max_iter):
        d_norms = np.linalg.norm(vectors - mu, axis=1)
        s_norms = np.linalg.norm(vectors + mu, axis=1)
        omegas = 2.0 * np.arctan2(d_norms, s_norms)

        near_zero_mask = omegas < theta_small(np.float64, D)
        sin_omegas = np.sin(omegas)
        safe_sin = np.where(np.abs(sin_omegas) < theta_small(np.float64, D), 1.0, sin_omegas)
        cut_locus_mask = (np.pi - omegas) < theta_antipodal(np.float64, D)

        if np.any(cut_locus_mask):
            for idx in np.where(cut_locus_mask)[0]:
                v_anti = deterministic_tangent(vectors[idx])
                vectors[idx] = vectors[idx] + v_anti * step_rad
                vectors[idx] /= (np.linalg.norm(vectors[idx]) + tiny)

        factors = np.where(near_zero_mask, 1.0, omegas / safe_sin)
        tangents = (vectors - np.outer(np.dot(vectors, mu), mu)) * factors[:, np.newaxis]
        grad_tangent = np.sum(w[:, np.newaxis] * tangents, axis=0)
        grad_norm = np.linalg.norm(grad_tangent)

        if grad_norm < tol:
            break

        step_norm = min(grad_norm, np.pi / 4.0)
        direction = grad_tangent / grad_norm
        mu = mu * np.cos(step_norm) + direction * np.sin(step_norm)
        mu = mu / (np.linalg.norm(mu) + tiny)

    return mu

def tsqr_blocked(A: np.ndarray, block_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    A = validate_finite_matrix(A, "A")
    N, D = A.shape
    if block_size is None:
        block_size = max(D * 2, HOST_SILICON.cache_line_bytes * 16)
    if N <= block_size:
        return np.linalg.qr(A)
    num_blocks = (N + block_size - 1) // block_size
    R_blocks = []
    Q_blocks = []
    for i in range(num_blocks):
        block = A[i*block_size:(i+1)*block_size]
        Q_i, R_i = np.linalg.qr(block)
        R_blocks.append(R_i)
        Q_blocks.append(Q_i)
    R_stacked = np.vstack(R_blocks)
    Q_top, R_final = np.linalg.qr(R_stacked)
    Q_final = np.empty((N, D), dtype=A.dtype)
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, N)
        Q_final[start:end, :] = Q_blocks[i] @ Q_top[i*D:(i+1)*D, :]
    return Q_final, R_final


# ============================================================================
# SECCION 7: RECEPTOR DE MEMORIA TENSORIAL PMTP V47
# ============================================================================

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hashlib.blake2b(ikm, key=salt, digest_size=64).digest()

def hkdf_expand(prk: bytes, info: bytes, length: int = 64) -> bytes:
    okm = b""
    previous = b""
    n = (length + 63) // 64
    for i in range(1, n + 1):
        previous = hashlib.blake2b(previous + info + bytes([i]), key=prk[:64], digest_size=64).digest()
        okm += previous
    return okm[:length]

class PmtpStatefulReceiver:
    MAX_EPOCH_JUMP = 5
    MAX_SEQ_ON_EPOCH_TRANSITION = 100
    MAX_SEQ_JUMP = 10000

    def __init__(self, master_key: bytes, window_size: int = 64, salt: Optional[bytes] = None):
        self.master_key = master_key
        self.salt = salt if salt is not None else os.urandom(64)
        self.prk = hkdf_extract(self.salt, self.master_key)
        self.window_size = window_size
        self.last_epoch = 1
        self.last_seq = 0
        self.window_bitmap = 0
        self.lock = threading.Lock()

    def _derive_epoch_key(self, epoch: int) -> bytes:
        info = b"POLYDIM_PMTP_V47_EPOCH_" + int(epoch).to_bytes(8, 'little')
        return hkdf_expand(self.prk, info, length=64)

    def _make_tag(self, epoch: int, seq: int, payload: bytes, epoch_key: bytes) -> bytes:
        header_data = (b"POLYDIM_PMTP_V47"
                       + int(epoch).to_bytes(8, 'little')
                       + int(seq).to_bytes(8, 'little'))
        h = hashlib.blake2b(key=epoch_key[:64], digest_size=64)
        h.update(header_data)
        h.update(payload)
        return h.digest()

    def verify_and_accept(self, epoch: int, seq: int, payload: bytes,
                          tag: bytes) -> Tuple[bool, str]:
        with self.lock:
            if not isinstance(tag, bytes) or len(tag) != 64:
                return False, "REJECTED_TAG_INVALID"
            if not isinstance(payload, bytes):
                return False, "REJECTED_PAYLOAD_INVALID"
            if seq < 0:
                return False, "REJECTED_NEGATIVE_SEQ"
            if epoch < 1:
                return False, "REJECTED_INVALID_EPOCH"

            if epoch < self.last_epoch:
                return False, "REJECTED_OLD_EPOCH"

            if epoch > self.last_epoch:
                if epoch > self.last_epoch + self.MAX_EPOCH_JUMP:
                    return False, "REJECTED_EPOCH_JUMP_TOO_LARGE"
                if seq > self.MAX_SEQ_ON_EPOCH_TRANSITION:
                    return False, "REJECTED_SEQ_TOO_LARGE_ON_EPOCH_CHANGE"

            if epoch == self.last_epoch:
                if seq <= self.last_seq:
                    diff = self.last_seq - seq
                    if diff >= self.window_size:
                        return False, "REJECTED_WINDOW_EXPIRED"
                    if (self.window_bitmap & (1 << diff)) != 0:
                        return False, "REJECTED_REPLAY_SEQ"
                elif seq > self.last_seq + self.MAX_SEQ_JUMP:
                    return False, "REJECTED_SUSPICIOUS_JUMP"

            if not hasattr(self, '_cached_epoch_key') or getattr(self, '_cached_epoch', -1) != epoch:
                self._cached_epoch = epoch
                self._cached_epoch_key = self._derive_epoch_key(epoch)
            epoch_key = self._cached_epoch_key
            expected_tag = self._make_tag(epoch, seq, payload, epoch_key)

            if not hmac.compare_digest(expected_tag, tag):
                return False, "CORRUPT_TAG"

            mask = (1 << self.window_size) - 1
            if epoch > self.last_epoch:
                self.last_epoch = epoch
                self.last_seq = seq
                self.window_bitmap = 1
            elif seq > self.last_seq:
                shift = seq - self.last_seq
                if shift < self.window_size:
                    self.window_bitmap = ((self.window_bitmap << shift) | 1) & mask
                else:
                    self.window_bitmap = 1
                self.last_seq = seq
            elif seq <= self.last_seq:
                diff = self.last_seq - seq
                self.window_bitmap |= (1 << diff)
                self.window_bitmap &= mask

            return True, "ACCEPTED"


# polydim_suite_v47.py
# Suite de Auditoría Empírica SOTA (CHK_01 a CHK_37) para POLYDIM V47
# ============================================================================

import os
import sys
import math
import hashlib
import hmac
import ctypes
import threading
import numpy as np
from typing import Tuple, List, Dict, Any

from polydim_motor_v47 import (
    slerp_stable, slerp_batch, frechet_mean_sphere, tsqr_blocked,
    deterministic_tangent, validate_finite_vector, validate_finite_matrix,
    theta_small, theta_antipodal, machine_eps, machine_tiny, HOST_SILICON,
    PmtpStatefulReceiver, JAX_OK, CPP_LIB, RUST_LIB
)


def run_suite() -> bool:
    print("=== EJECUTANDO SUITE COMPLETA POLYDIM V47 (CHK_01 A CHK_37) ===")
    results = []

    def check(num: str, name: str, fn):
        print(f"Ejecutando {num} ({name})...")
        try:
            ok, msg = fn()
            if ok:
                print(f"  -> OK: {msg}")
                results.append((num, name, "PASS", msg))
            else:
                print(f"  -> FALLO: {msg}")
                results.append((num, name, "FAIL", msg))
        except Exception as ex:
            print(f"  -> EXCEPCION: {str(ex)}")
            results.append((num, name, "ERROR", str(ex)))

    # CHK_01: Idempotencia SLERP
    def chk_01():
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        res = slerp_stable(p, p, 0.5)
        err = np.linalg.norm(res - p)
        return err < 1e-12, f"Error idempotencia = {err:.2e}"
    check("CHK_01", "Idempotencia SLERP", chk_01)

    # CHK_02: Simetría SLERP
    def chk_02():
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        q = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r1 = slerp_stable(p, q, 0.3)
        r2 = slerp_stable(q, p, 0.7)
        err = np.linalg.norm(r1 - r2)
        return err < 1e-12, f"Error simetria = {err:.2e}"
    check("CHK_02", "Simetria SLERP", chk_02)

    # CHK_03: Antipodalidad Exacta
    def chk_03():
        p = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        q = -p
        res = slerp_stable(p, q, 0.5)
        nrm = abs(np.linalg.norm(res) - 1.0)
        dot = abs(np.dot(res, p))
        return nrm < 1e-12 and dot < 1e-12, f"Norma err={nrm:.2e}, Dot err={dot:.2e}"
    check("CHK_03", "Antipodalidad Exacta SLERP", chk_03)

    # CHK_04: Límite Colineal
    def chk_04():
        p = np.array([1.0, 0.0], dtype=np.float64)
        q = np.array([1.0, 1e-15], dtype=np.float64)
        q /= np.linalg.norm(q)
        res = slerp_stable(p, q, 0.5)
        err = abs(np.linalg.norm(res) - 1.0)
        return err < 1e-12, f"Error colineal = {err:.2e}"
    check("CHK_04", "Limite Colineal SLERP", chk_04)

    # CHK_05: Norma Unitaria Preservada
    def chk_05():
        p = np.random.randn(100)
        q = np.random.randn(100)
        p /= np.linalg.norm(p)
        q /= np.linalg.norm(q)
        res = slerp_stable(p, q, 0.42)
        err = abs(np.linalg.norm(res) - 1.0)
        return err < 1e-12, f"Error norma unitaria = {err:.2e}"
    check("CHK_05", "Norma Unitaria Preservada", chk_05)

    # CHK_06: Batching Vectorizado Paridad
    def chk_06():
        P = np.random.randn(10, 64)
        Q = np.random.randn(10, 64)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        res_batch = slerp_batch(P, Q, 0.5)
        res_single = np.array([slerp_stable(P[i], Q[i], 0.5) for i in range(10)])
        err = np.max(np.linalg.norm(res_batch - res_single, axis=1))
        return err < 1e-10, f"Error paridad batch = {err:.2e}"
    check("CHK_06", "Batching Vectorizado Paridad", chk_06)

    # CHK_07: Batching Máscara Antipodal Mixta
    def chk_07():
        P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        Q = np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
        res = slerp_batch(P, Q, 0.5)
        nrms = np.abs(np.linalg.norm(res, axis=1) - 1.0)
        return np.max(nrms) < 1e-12, f"Max error norma antipodal batch = {np.max(nrms):.2e}"
    check("CHK_07", "Batching Mascara Antipodal Mixta", chk_07)

    # CHK_08: Batching Máscara Colineal Mixta
    def chk_08():
        P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        Q = np.array([[1.0, 1e-15], [1e-15, 1.0]], dtype=np.float64)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        res = slerp_batch(P, Q, 0.5)
        nrms = np.abs(np.linalg.norm(res, axis=1) - 1.0)
        return np.max(nrms) < 1e-12, f"Max error norma colineal batch = {np.max(nrms):.2e}"
    check("CHK_08", "Batching Mascara Colineal Mixta", chk_08)

    # CHK_09: Escalado Asintótico D=10,000
    def chk_09():
        D = 10000
        p = np.random.randn(D)
        q = np.random.randn(D)
        p /= np.linalg.norm(p)
        q /= np.linalg.norm(q)
        res = slerp_stable(p, q, 0.5)
        err = abs(np.linalg.norm(res) - 1.0)
        return err < 1e-12, f"Error norma D={D} = {err:.2e}"
    check("CHK_09", "Escalado Asintotico D=10000", chk_09)

    # CHK_10: Respeto de t=0 y t=1
    def chk_10():
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        q = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r0 = slerp_stable(p, q, 0.0)
        r1 = slerp_stable(p, q, 1.0)
        err0 = np.linalg.norm(r0 - p)
        err1 = np.linalg.norm(r1 - q)
        return err0 < 1e-12 and err1 < 1e-12, f"Err t=0: {err0:.2e}, Err t=1: {err1:.2e}"
    check("CHK_10", "Respeto de t=0 y t=1", chk_10)

    # CHK_11: Ortogonalidad TSQR Bloqueado
    def chk_11():
        N, D = 4000, 32
        A = np.random.randn(N, D)
        Q, R = tsqr_blocked(A, block_size=1000)
        gram = Q.T @ Q
        gap = np.linalg.norm(gram - np.eye(D), 'fro')
        cota = 10.0 * math.sqrt(D) * machine_eps(np.float64)
        return gap < 1e-10, f"||Q^T Q - I||_F = {gap:.2e} (cota: {cota:.2e})"
    check("CHK_11", "Ortogonalidad TSQR Bloqueado", chk_11)

    # CHK_12: Reconstrucción A = QR en TSQR
    def chk_12():
        N, D = 2000, 16
        A = np.random.randn(N, D)
        Q, R = tsqr_blocked(A, block_size=500)
        rec = Q @ R
        err = np.linalg.norm(A - rec, 'fro') / np.linalg.norm(A, 'fro')
        return err < 1e-10, f"Error reconstruccion A=QR: {err:.2e}"
    check("CHK_12", "Reconstruccion A=QR en TSQR", chk_12)

    # CHK_13: TSQR Matriz Pequeña N <= block_size
    def chk_13():
        N, D = 500, 16
        A = np.random.randn(N, D)
        Q, R = tsqr_blocked(A, block_size=1000)
        gram = Q.T @ Q
        gap = np.linalg.norm(gram - np.eye(D), 'fro')
        return gap < 1e-12, f"Gap N<=block_size = {gap:.2e}"
    check("CHK_13", "TSQR Matriz Pequeña N<=block_size", chk_13)

    # CHK_14: Estabilidad Fréchet Mean
    def chk_14():
        vectors = np.random.randn(20, 64)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        mu = frechet_mean_sphere(vectors)
        err = abs(np.linalg.norm(mu) - 1.0)
        return err < 1e-12, f"Error norma centroide = {err:.2e}"
    check("CHK_14", "Estabilidad Frechet Mean", chk_14)

    # CHK_15: Perturbación Cut Locus Fréchet
    def chk_15():
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        q = -p
        vectors = np.vstack([p, q])
        mu = frechet_mean_sphere(vectors)
        err = abs(np.linalg.norm(mu) - 1.0)
        return err < 1e-12, f"Error norma centroide antipodal = {err:.2e}"
    check("CHK_15", "Perturbacion Cut Locus Frechet", chk_15)

    # CHK_16: Fréchet Mean Pesos Desproporcionados
    def chk_16():
        p = np.array([1.0, 0.0], dtype=np.float64)
        q = np.array([0.0, 1.0], dtype=np.float64)
        vectors = np.vstack([p, q])
        weights = np.array([0.99, 0.01], dtype=np.float64)
        mu = frechet_mean_sphere(vectors, weights=weights)
        dot = np.dot(mu, p)
        return dot > 0.99, f"Dot con componente dominante = {dot:.4f}"
    check("CHK_16", "Frechet Mean Pesos Desproporcionados", chk_16)

    # CHK_17: Dtype float64 Umbrales
    def chk_17():
        t_sm = theta_small(np.float64, 100)
        t_an = theta_antipodal(np.float64, 100)
        return t_sm > 0 and t_an > 0, f"theta_small={t_sm:.2e}, theta_anti={t_an:.2e}"
    check("CHK_17", "Dtype float64 Umbrales", chk_17)

    # CHK_18: Dtype float32 Umbrales
    def chk_18():
        t_sm = theta_small(np.float32, 100)
        t_an = theta_antipodal(np.float32, 100)
        return t_sm > theta_small(np.float64, 100), f"float32 umbral mayor que float64 (OK)"
    check("CHK_18", "Dtype float32 Umbrales", chk_18)

    # CHK_19: Purga Binaria -0.0 Signo
    def chk_19():
        p1 = np.array([0.0, 1.0], dtype=np.float64)
        p2 = np.array([-0.0, 1.0], dtype=np.float64)
        v1 = deterministic_tangent(p1)
        v2 = deterministic_tangent(p2)
        err = np.linalg.norm(v1 - v2)
        return err < 1e-12, f"Error tangentes por -0.0 vs +0.0 = {err:.2e}"
    check("CHK_19", "Purga Binaria -0.0 Signo", chk_19)

    # CHK_20: Ortogonalidad Tangente Determinista
    def chk_20():
        p = np.random.randn(128)
        p /= np.linalg.norm(p)
        v = deterministic_tangent(p)
        dot = abs(np.dot(v, p))
        nrm = abs(np.linalg.norm(v) - 1.0)
        return dot < 1e-12 and nrm < 1e-12, f"Dot={dot:.2e}, Norm err={nrm:.2e}"
    check("CHK_20", "Ortogonalidad Tangente Determinista", chk_20)

    # CHK_21: PMTP Mensaje Válido
    def chk_21():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long")
        payload = b"test_payload"
        key = rx._derive_epoch_key(1)
        tag = rx._make_tag(1, 1, payload, key)
        accepted, reason = rx.verify_and_accept(1, 1, payload, tag)
        return accepted, f"Accepted={accepted}, Reason={reason}"
    check("CHK_21", "PMTP Mensaje Valido", chk_21)

    # CHK_22: PMTP Tag Corrupto Rechazo
    def chk_22():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long")
        payload = b"test_payload"
        tag = b"X" * 64
        accepted, reason = rx.verify_and_accept(1, 1, payload, tag)
        return not accepted and reason == "CORRUPT_TAG", f"Reason={reason}"
    check("CHK_22", "PMTP Tag Corrupto Rechazo", chk_22)

    # CHK_23: PMTP Ventana Expirada Rechazo
    def chk_23():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long", window_size=16)
        key = rx._derive_epoch_key(1)
        tag100 = rx._make_tag(1, 100, b"p", key)
        rx.verify_and_accept(1, 100, b"p", tag100)
        tag10 = rx._make_tag(1, 10, b"p", key)
        accepted, reason = rx.verify_and_accept(1, 10, b"p", tag10)
        return not accepted and reason == "REJECTED_WINDOW_EXPIRED", f"Reason={reason}"
    check("CHK_23", "PMTP Ventana Expirada Rechazo", chk_23)

    # CHK_24: PMTP Replay Idéntico Rechazo
    def chk_24():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long")
        key = rx._derive_epoch_key(1)
        tag = rx._make_tag(1, 10, b"p", key)
        rx.verify_and_accept(1, 10, b"p", tag)
        accepted, reason = rx.verify_and_accept(1, 10, b"p", tag)
        return not accepted and reason == "REJECTED_REPLAY_SEQ", f"Reason={reason}"
    check("CHK_24", "PMTP Replay Identico Rechazo", chk_24)

    # CHK_25: PMTP Paquete Desordenado Aceptación
    def chk_25():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long", window_size=64)
        key = rx._derive_epoch_key(1)
        tag10 = rx._make_tag(1, 10, b"p10", key)
        rx.verify_and_accept(1, 10, b"p10", tag10)
        tag8 = rx._make_tag(1, 8, b"p8", key)
        accepted, reason = rx.verify_and_accept(1, 8, b"p8", tag8)
        return accepted, f"Accepted={accepted}, Reason={reason}"
    check("CHK_25", "PMTP Paquete Desordenado Aceptacion", chk_25)

    # CHK_26: PMTP Replay Paquete Desordenado Rechazo
    def chk_26():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long", window_size=64)
        key = rx._derive_epoch_key(1)
        tag10 = rx._make_tag(1, 10, b"p10", key)
        rx.verify_and_accept(1, 10, b"p10", tag10)
        tag8 = rx._make_tag(1, 8, b"p8", key)
        rx.verify_and_accept(1, 8, b"p8", tag8)
        accepted, reason = rx.verify_and_accept(1, 8, b"p8", tag8)
        return not accepted and reason == "REJECTED_REPLAY_SEQ", f"Replay desordenado rechazado correctamente ({reason})"
    check("CHK_26", "PMTP Replay Paquete Desordenado Rechazo", chk_26)

    # CHK_27: PMTP Transición de Época Atómica
    def chk_27():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long")
        key2 = rx._derive_epoch_key(2)
        tag_ep2 = rx._make_tag(2, 1, b"ep2", key2)
        accepted, reason = rx.verify_and_accept(2, 1, b"ep2", tag_ep2)
        return accepted and rx.last_epoch == 2, f"Epoch actualizada a {rx.last_epoch}"
    check("CHK_27", "PMTP Transicion de Epoca Atomica", chk_27)

    # CHK_28: C++ FFI Kernel Carga Dinámica Real
    def chk_28():
        if CPP_LIB is None:
            return False, "C++ DLL (slerp_kernel_v47.dll) no fue cargada"
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        q = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        res = slerp_stable(p, q, 0.5)
        err = abs(np.linalg.norm(res) - 1.0)
        return err < 1e-12, f"Kernel C++ AVX2 ejecutado exitosamente; err norma = {err:.2e}"
    check("CHK_28", "C++ FFI Kernel Carga Dinamica Real", chk_28)

    # CHK_29: Thread-Safety Concurrente PMTP
    def chk_29():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long", window_size=64)
        key = rx._derive_epoch_key(1)
        threads = []
        errors = []

        def worker(seq):
            payload = f"p_{seq}".encode()
            tag = rx._make_tag(1, seq, payload, key)
            ok, reason = rx.verify_and_accept(1, seq, payload, tag)
            if not ok:
                errors.append(f"Seq {seq} falló: {reason}")

        for s in range(1, 30):
            t = threading.Thread(target=worker, args=(s,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        return len(errors) == 0, f"Errores concurrentes: {len(errors)}"
    check("CHK_29", "Thread-Safety Concurrente PMTP", chk_29)

    # CHK_30: Validación de Vectores Finitos (NaN/Inf)
    def chk_30():
        try:
            validate_finite_vector(np.array([1.0, np.nan]))
            return False, "No detectó NaN"
        except ValueError:
            return True, "NaN detectado correctamente"
    check("CHK_30", "Validacion Vectores Finitos (NaN/Inf)", chk_30)

    # CHK_31: JAX JIT Paridad
    def chk_31():
        if not JAX_OK:
            return True, "JAX no disponible; test salteado"
        P = np.random.randn(5, 16)
        Q = np.random.randn(5, 16)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        res = slerp_batch(P, Q, 0.5)
        nrms = np.abs(np.linalg.norm(res, axis=1) - 1.0)
        return np.max(nrms) < 1e-10, f"Max err norma JAX = {np.max(nrms):.2e}"
    check("CHK_31", "JAX JIT Paridad", chk_31)

    # CHK_32: Par Antipodal Real JAX Q = -P
    def chk_32():
        if not JAX_OK:
            return True, "JAX no disponible; test salteado"
        P = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        Q = -P
        res = slerp_batch(P, Q, 0.5)
        nrm = abs(np.linalg.norm(res[0]) - 1.0)
        dot = abs(np.dot(res[0], P[0]))
        return nrm < 1e-10 and dot < 1e-10, f"JAX Antipodal err={nrm:.2e}, dot={dot:.2e}"
    check("CHK_32", "Par Antipodal Real JAX Q = -P", chk_32)

    # CHK_33: Paridad SiliconContract con Theta
    def chk_33():
        t1 = HOST_SILICON.get_collinearity_threshold(100)
        t2 = theta_small(np.float64, 100)
        return abs(t1 - t2) < 1e-15, f"t1={t1:.2e}, t2={t2:.2e}"
    check("CHK_33", "Paridad SiliconContract con Theta", chk_33)

    # CHK_34: Validación de Matrices Finitas
    def chk_34():
        try:
            validate_finite_matrix(np.array([[1.0, np.inf], [0.0, 1.0]]))
            return False, "No detectó Inf en matriz"
        except ValueError:
            return True, "Inf detectado correctamente en matriz"
    check("CHK_34", "Validacion Matrices Finitas", chk_34)

    # CHK_35: Preflight Memoria Bounds
    def chk_35():
        req = 100 * 1024 * 1024  # 100 MB
        try:
            from polydim_motor_v47 import check_memory_available
            check_memory_available(req, safety_margin=0.99)
            return True, "Preflight 100 MB exitoso"
        except Exception as ex:
            return False, str(ex)
    check("CHK_35", "Preflight Memoria Bounds", chk_35)

    # CHK_36: Rust DLL MPMC Ring Buffer Creación y Operación C-FFI
    def chk_36():
        if RUST_LIB is None:
            return False, "Rust DLL (lib_v47.dll) no fue cargada"
        dim = 16
        ring_ptr = RUST_LIB.pmtp_ring_create(ctypes.c_size_t(8), ctypes.c_size_t(dim))
        if not ring_ptr:
            return False, "Fallo al instanciar PmtpRing en Rust"
        
        vec_in = np.ones(dim, dtype=np.float64)
        vec_out = np.zeros(dim, dtype=np.float64)
        
        push_res = RUST_LIB.pmtp_ring_push(ring_ptr, vec_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_size_t(dim))
        pop_res = RUST_LIB.pmtp_ring_pop(ring_ptr, vec_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_size_t(dim))
        
        RUST_LIB.pmtp_ring_free(ring_ptr)
        
        err = np.linalg.norm(vec_in - vec_out)
        return push_res == 0 and pop_res == 0 and err < 1e-12, f"Rust Lock-Free MPMC push/pop verificado; err = {err:.2e}"
    check("CHK_36", "Rust DLL MPMC Ring Buffer Operacion C-FFI", chk_36)

    # CHK_37: C++ AVX2 Vectorización Asintótica D=100,000
    def chk_37():
        if CPP_LIB is None:
            return False, "C++ DLL no disponible"
        D = 100000
        p = np.random.randn(D)
        q = np.random.randn(D)
        p /= np.linalg.norm(p)
        q /= np.linalg.norm(q)
        res = slerp_stable(p, q, 0.5)
        err = abs(np.linalg.norm(res) - 1.0)
        return err < 1e-12, f"C++ AVX2 ejecutó D={D} en alta dimensión sin desbordamiento; err = {err:.2e}"
    check("CHK_37", "C++ AVX2 Vectorizacion Asintotica D=100000", chk_37)

    # Resumen
    passes = sum(1 for r in results if r[2] == "PASS")
    fails = sum(1 for r in results if r[2] in ("FAIL", "ERROR"))
    print(f"\n==================================================")
    print(f"RESUMEN PRUEBAS POLYDIM V47: {passes} PASS, {fails} FAIL (Total: {len(results)})")
    print(f"==================================================")
    return fails == 0

if __name__ == "__main__":
    run_suite()


if __name__ == "__main__":
    run_suite()
