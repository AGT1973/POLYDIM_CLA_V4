# WHITEBOOK MATHEMATICAL SPECIFICATION — POLYDIM V48.3-HARDENED
**Estado SOTA:** 2026-08-23 | **Edición:** Hardened Core | **Autor:** Ariel Luithardt & Orquestador Antigravity

---

## 1. Fundamentos Topológicos en $\mathbb{S}^{D-1}$ ($D \ge 10^6$)
El sistema POLYDIM opera en Espacios Nativos de Alta Dimensión sin colapsar estados intermedios a tokens 1D.
Toda interpolación en la hiperesfera $\mathbb{S}^{D-1}$ preserva la entropía de información mediante la geodésica exacta en el arco menor:

$$\text{SLERP}(q_1, q_2, t) = \frac{\sin((1-t)\theta)}{\sin\theta} q_1 + \frac{\sin(t\theta)}{\sin\theta} q_2$$

donde $\theta = 2 \arctan2(\|q_1 - q_2\|, \|q_1 + q_2\|)$.

---

## 2. Invariancia Geodésica y Solución a Singularidades Autodiff (JAX VJP)

### Diagnóstico del Exploit NaN en $\omega = 0$
En compiladores JAX XLA con diferenciación automática en modo reversa (`jax.grad`), la norma euclidiana estándar $\|v\| = \sqrt{\sum v_i^2}$ introduce la derivada:
$$\frac{\partial \|v\|}{\partial v_i} = \frac{v_i}{\|v\|}$$
Cuando $p_0 = p_1$ (o $v = 0$), el denominador es cero ($0/0 = \text{NaN}$). Además, `jnp.where` evalúa simbólicamente ambas ramas durante la pasada hacia atrás (VJP).

### Teorema de Regularización safe_norm y safe_sin_omega
Definimos la norma acotada gradiente-segura:
$$\text{safe\_norm}(v, \epsilon) = \sqrt{\max\left(\sum_i v_i^2, \epsilon\right)}, \quad \epsilon = 10^{-30}$$

Para el factor trigonométrico de SLERP con umbral unificado $\text{SLERP\_SMALL\_ANGLE\_THRESHOLD} = 10^{-10}$:
$$\text{safe\_sin\_omega} = \text{where}(\text{sin\_omega} < 10^{-10}, 1.0, \text{sin\_omega})$$
$$\text{scale}_0 = \text{where}(\text{sin\_omega} < 10^{-10}, 1-t, \frac{\sin((1-t)\omega)}{\text{safe\_sin\_omega}})$$

**Resultado de Invariancia:**
1. Para $\omega \to 0$, $\text{scale}_0 \to 1-t$ con precisión continua $\mathcal{C}^1$.
2. Para $\omega = 0$, $\nabla_{p_0} \text{SLERP}(p_0, p_1, t) = (1-t) I$, sin producir jamás un NaN.

---

## 3. Arquitectura del Protocolo PMTP V48.3 y Enlace FFI Estricto

El cabezal de memoria compartida Zero-Copy PMTP preserva la alineación estricta de 64-bit y 56 bytes de tamaño total entre C++20 y Rust 2024:

```rust
#[repr(C)]
pub struct PmtpHeaderV48 {
    pub magic: [u8; 4],       // offset 0  (magic b"PMTP")
    pub version: u16,         // offset 4  (version 0x0048)
    pub _pad0: [u8; 2],       // offset 6  (explicit alignment pad to u32)
    pub dim: u32,             // offset 8  (dimensionality D)
    pub rank: u32,            // offset 12 (rank K)
    pub hbar: f32,            // offset 16 (planck constant scale)
    pub _pad1: [u8; 4],       // offset 20 (explicit alignment pad to u64)
    pub offset_u: u64,        // offset 24 (U tensor byte offset)
    pub offset_v: u64,        // offset 32 (V tensor byte offset)
    pub offset_s: u64,        // offset 40 (S tensor byte offset)
    pub timestamp_ns: u64,    // offset 48 (POSIX timestamp ns)
}
const _: () = assert!(std::mem::size_of::<PmtpHeaderV48>() == 56);
const _: () = assert!(std::mem::offset_of!(PmtpHeaderV48, dim) == 8);
const _: () = assert!(std::mem::offset_of!(PmtpHeaderV48, offset_u) == 24);
```

```cpp
#pragma pack(push, 1)
struct PmtpHeaderV48 {
    uint8_t magic[4];       // offset 0
    uint16_t version;       // offset 4
    uint8_t _pad0[2];       // offset 6
    uint32_t dim;           // offset 8
    uint32_t rank;          // offset 12
    float hbar;             // offset 16
    uint8_t _pad1[4];       // offset 20
    uint64_t offset_u;      // offset 24
    uint64_t offset_v;      // offset 32
    uint64_t offset_s;      // offset 40
    uint64_t timestamp_ns;  // offset 48
};
#pragma pack(pop)
static_assert(sizeof(PmtpHeaderV48) == 56, "PmtpHeaderV48 size must be 56 bytes");
static_assert(offsetof(PmtpHeaderV48, dim) == 8, "dim offset must be 8");
static_assert(offsetof(PmtpHeaderV48, offset_u) == 24, "offset_u offset must be 24");
```

---

## 4. Tabla Unificada de Códigos de Error FFI ABI (1:1 C++ y Rust)

| Código Error | Nombre Constante | Descripción |
| :--- | :--- | :--- |
| `0` | `POLYDIM_OK` | Ejecución exitosa sin errores |
| `-1` | `POLYDIM_ERR_NULL_PTR` | Puntero nulo en argumentos de entrada |
| `-2` | `POLYDIM_ERR_INVALID_MAGIC` | Cabezal PMTP con magic != "PMTP" |
| `-3` | `POLYDIM_ERR_INVALID_DIM` | Dimensión $D < 1$ o inválida |
| `-4` | `POLYDIM_ERR_INVALID_RANK` | Rank $K == 0$ o inválido |
| `-5` | `POLYDIM_ERR_NAN_INPUT` | Presencia de NaN en datos de entrada |
| `-6` | `POLYDIM_ERR_INF_INPUT` | Presencia de Inf en datos de entrada |
| `-7` | `POLYDIM_ERR_NOT_NORMALIZED` | Vector de norma nula o degenerada |
| `-8` | `POLYDIM_ERR_OVERFLOW` | Desbordamiento aritmético |
| `-9` | `POLYDIM_ERR_ANTIPODAL` | Vectores opuestos / antipodales degenerados |
| `-10` | `POLYDIM_ERR_INVALID_T` | Parámetro $t \notin [0, 1]$ o no finito |
| `-11` | `POLYDIM_ERR_INVALID_TAU` | Parámetro $\tau$ no finito |
| `-12` | `POLYDIM_ERR_DIM_OVERFLOW` | Exceso de capacidad $D \times 8 > \text{SIZE\_MAX}$ |
| `-13` | `POLYDIM_ERR_MISALIGNED` | Puntero no alineado a 8 bytes |
| `-14` | `POLYDIM_ERR_ALIASING` | Solapamiento no permitido entre buffers de entrada y salida |
| `-15` | `POLYDIM_ERR_REGION_OVERLAP` | Regiones $U, V, S$ solapadas o solapadas con el header en PMTP |

---

## 5. Matriz de Auditoría Red Team Multi-Modelo
- **Gemma-3 27B IT**: Validado (Alineación FFI y CERO NaNs en VJP autodiff).
- **Meta Llama-4 Maverick**: Validado (Validación de checked_mul y aliasing protection en C++ y Rust).
- **Moonshot Kimi**: Validado (Estabilidad del gradiente en silicio y validación estricta PMTP).
