# WHITEBOOK — POLYDIM EINSOF V40
## Arquitectura Tri-Núcleo (JAX + Rust + C++)

### Filosofía
POLYDIM opera en Espacios Nativos de Alta Dimension (ND >= 100,000).
Colapsar pensamiento ND a tokens 1D en cada paso destruye entropía geométrica.
Los agentes se comunican via tensores nativos (Protocolo PMTP), no texto serializado.

---

## Diagrama de Capas

```
┌─────────────────────────────────────────────┐
│  CAPA 1: einsof_jax (Investigación/GPU)     │
│  • grad(slerp), jit(tsqr), vmap(batch)      │
│  • Entrenamiento en GPU/TPU                 │
│  • Diferenciación automática                │
│  Archivos: slerp.py, tsqr.py, stiefel.py   │
└──────────────────┬──────────────────────────┘
                   │ C extension (PyO3)
┌──────────────────▼──────────────────────────┐
│  CAPA 2: einsof_cpp (Kernel de Producción)  │
│  • BLAS/LAPACK nativo, SIMD, OpenMP         │
│  • std::hardware_destructive_interference_size│
│  • CholeskyQR2 + TSQR con Eigen             │
│  Archivos: slerp_kernel.cpp, tsqr.cpp       │
└──────────────────┬──────────────────────────┘
                   │ FFI seguro
┌──────────────────▼──────────────────────────┐
│  CAPA 3: einsof_rust (Bus PMTP + Seguridad) │
│  • Ring buffer lock-free GARANTIZADO        │
│  • HMAC-BLAKE2b keyed + epoch anti-replay   │
│  • Sin data races: borrow checker en compile│
│  Archivos: lib.rs, pmtp_bus.rs, hmac.rs     │
└─────────────────────────────────────────────┘
```

---

## Por qué 3 lenguajes y no uno

| Problema | JAX | Rust | C++ |
|---|---|---|---|
| grad(slerp) automático | ÚNICO | ❌ | ❌ |
| GPU/TPU sin CUDA manual | ÚNICO | ❌ | ❌ |
| Ring buffer sin data races garantizadas | ❌ | ÚNICO | ⚠️ UB silencioso |
| BLAS/Eigen nativo + SIMD estable | ❌ | ⚠️ FFI | ÚNICO |
| std::hardware_destructive_interference_size | ❌ | ❌ | ÚNICO |

---

## Invariantes matemáticos certificados (15 CHKs PASS)

| CHK | Invariante | Resultado |
|---|---|---|
| 01 | SLERP atan2 geodésico D=100K | drift 2.98e-7 |
| 02 | Antipodal sin NaN | omega=π exacto |
| 03 | PMTP ring coherencia | 100/100 writes |
| 04 | HMAC 1-bit detection | 4.6e-10 delta detectado |
| 05 | Silicon probe (sin hardcoding) | page=4096, ram=11.38GB |
| 06 | Tree-TSQR D=100K K=64 | cond=1.048, 24.41MB |
| 07 | Stiefel drift 5K pasos | drift_max=2.38e-7 |
| 08 | SLERP sum/diff ratio | drift=1.78e-7 |
| 09 | Antipodal BLAKE2b escape | ortho=9.5e-18 |
| 10 | Drift sqrt(N) no lineal | regime=sqrt(N) |
| 11 | TSQR gap dinámico | gap_after=0.0 |
| 12 | CholeskyQR2 kappa guard | gap=1.37e-15 |
| 13 | False sharing guard | separation=64B |
| 14 | HMAC anti-replay epoch | epoch+seq rechazado |
| 15 | f64 obligatorio geometría | error f32 cuantificado |

---

## Reglas de oro del triple núcleo

1. **bfloat16/f32**: solo almacenamiento de embeddings
2. **f64**: obligatorio para normas, ángulos, SLERP, reortogonalización
3. **Ningún parámetro de silicio hardcodeado** — todo derivado en runtime
4. **TSQR por defecto**; CholeskyQR2 solo si kappa(A) < sqrt(1/eps_f32) ~ 4096
5. **PMTP**: head y tail en cache lines distintas — padding obligatorio
6. **HMAC**: siempre epoch + seq_number — anti-replay estricto
7. **Antipodal**: tangente determinista via BLAKE2b — no ruido aleatorio

## ⚠️ DOGMA INVIOLABLE: EL PROTOCOLO "NO-GUSANO" (FFI PMTP)

**Aviso para todo Agente / Sabueso / Tribunal:**
Está TERMINANTEMENTE PROHIBIDO sugerir, implementar o tolerar mecanismos de serialización 1D (JSON, Base64, REST, gRPC) para la transferencia de tensores entre el núcleo JAX y el núcleo Rust.

1. **La Tragedia de la Serialización:** Colapsar un espacio ND a texto destruye entropía y ciclos de CPU (Desigualdad DPI).
2. **Puente Nativo FFI:** La comunicación entre JAX y Rust se realiza **únicamente** a través de la interfaz C-ABI nativa (PyO3). 
3. **Mecanismo Certificado:** Los tensores se inyectan crudos en la memoria (como `Vec<f32>` o `numpy arrays`) a través de la frontera FFI, aterrizando directamente en el `PmtpRing` (Ring Buffer Lock-free de Rust).

Si intentas proponer un puerto TCP con JSON para esto, serás terminado.
