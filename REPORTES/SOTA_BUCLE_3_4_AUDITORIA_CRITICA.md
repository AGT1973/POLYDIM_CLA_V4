# COMPENDIO SOTA — BUCLES 3 Y 4 DE AUDITORÍA ADVERSARIAL (POLYDIM V78)

**Ubicación:** `E:\POLYDIM_EINSOF\REPORTES\SOTA_BUCLE_3_4_AUDITORIA_CRITICA.md`  
**Fecha:** 28 de Agosto de 2026  
**Auditor:** Red Team Externo (Auditoría Algebraica, Asintótica y Espectral)  
**Estado:** Ingesta Acumulada — Veto de Código Activo (Regla 19)

---

## 1. DESGLOSE DE HALLAZGOS CRÍTICOS (BUCLES 3 Y 4)

### 🔴 P0.1: Colapso Algebraico en `apply_spherical_rotor` ($\operatorname{rot}(x) = x$)
- **Mecanismo:** El código proyecta $U$ y $V$ al plano tangente $T_x S^{D-1}$ ($U_{\text{tan}} \perp x$, $V_{\text{tan}} \perp x$). Luego, tras ortonormalizar con `cholesky_qr3`, $u_{\text{orth}}, v_{\text{orth}} \in T_x S^{D-1}$.
- **Consecuencia:** $\operatorname{proj}_u(x) = \langle x, u_{\text{orth}} \rangle u_{\text{orth}} = 0$ y $\operatorname{proj}_v(x) = 0$.
- **Resultado:** La rotación degenera idénticamente en:
  $$\operatorname{rot}(x) = x - 0 - 0 + \cos(\theta) \cdot 0 + \sin(\theta) \cdot 0 = x$$
- **Falso Positivo:** Como $\|x\| = 1$, el test de norma $\|\operatorname{rot}(x)\| = 1.0$ pasa perfectamente, ocultando que el vector jamás se movió.

---

### 🔴 P0.2: Fórmula de Rotación 2D Incompleta
- **Diagnóstico:** Si $a = \langle x, u \rangle$ y $b = \langle x, v \rangle$, una rotación en el plano requiere:
  $$\begin{bmatrix} a' \\ b' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix}$$
- **Error en V78:** El código calculaba $a' = a\cos\theta$ y $b' = b\sin\theta$, omitiendo los términos cruzados y destruyendo la ortogonalidad intrínseca. La división final por `rot_norm` actuaba como "parche cosmético" para forzar norma 1.

---

### 🔴 P0.3: Divergencia Estructural en `shifted_cholqr3` vs Fukaya 2020
- **Algoritmo Real (Fukaya et al.):** 
  1. Paso 1: $R_1 = \operatorname{chol}(X^T X + s I)$, $Q_1 = X R_1^{-1}$ (Shifted).
  2. Paso 2: $R_2 = \operatorname{chol}(Q_1^T Q_1)$, $Q_2 = Q_1 R_2^{-1}$ (Sin shift).
  3. Paso 3: $R_3 = \operatorname{chol}(Q_2^T Q_2)$, $Q_3 = Q_2 R_3^{-1}$ (Sin shift).
  4. Reconstrucción: $R = R_3 R_2 R_1$.
- **Error en V78:** Aplicaba 3 veces el shift consecutivo en un bucle `for _ in range(3)`, sin devolver el factor triangular $R$, haciendo imposible verificar el error residual $\|A - QR\|_F / \|A\|_F$ (backward stability).

---

### 🔴 P0.4: Desajuste de Escala en el Shift de Tikhonov
- **Fórmula Teórica (Paper):** $s_{\text{paper}} = 11(mn + n(n+1)) u \|X\|_2^2$. Para $m=10^6, n=2$, $s \approx 2.2 \times 10^7 u$.
- **Fórmula V78:** $s_{\text{V78}} = 11 u \operatorname{tr}(G)/k \approx 11 u$.
- **Diferencia:** Una discrepancia de $\sim 2 \times 10^6$ en la magnitud del shift.

---

### 🔴 P0.5: Singularidad Antipodal en el Cut Locus
- **Diagnóstico:** El ángulo cordal $\theta = 2\operatorname{atan2}(\|x-y\|, \|x+y\|)$ es numéricamente estable, pero el vector unitario tangente $u = \frac{P_x y}{\|P_x y\|}$ colapsa a $0/0$ en el punto antipodal ($y = -x$).
- **Solución:** Declarar explícitamente una política de rama antipodal discontinua en lugar de afirmar falsamente que el gradiente es analíticamente suave en todo el espacio.

---

## 2. BLUEPRINT DE ARQUITECTURA PARA V79 (REFERENCE-FIRST)

```
                 REFERENCIA MATEMÁTICA PURA
                             │
              ┌──────────────┴──────────────┐
              ↓                             ↓
       ESFERA S^(D-1)                STIEFEL St(D,k)
    (Exp, Log Cordal)            (s-CholQR3, Wen-Yin Cayley)
              │                             │
              └──────────────┬──────────────┘
                             ↓
                   ORÁCULO DIFERENCIAL
                             │
                 ┌───────────┼───────────┐
                 ↓           ↓           ↓
               JAX          C++         Rust
                 └───────────┬───────────┘
                             ↓
                    PMTP v44 (199B / AEAD)
```

### Las 10 Pruebas Innegociables del Oráculo:
1. **Sphere Exp/Log Roundtrip:** $\operatorname{Exp}_x(\operatorname{Log}_x(y)) = y$.
2. **Sphere Identity:** $\operatorname{Log}_x(x) = 0$ con gradiente acotado.
3. **Sphere Near-Antipodal:** Fuzzing en $\theta \in [\pi - 10^{-2}, \pi - 10^{-12}]$.
4. **Antipodal Branching:** Verificación de selección de rama determinista.
5. **Rotor Movement:** $\|R(x) - x\| > 0$ cuando $\theta > 0$.
6. **Full Cayley vs SMW:** Equivalencia exacta entre matriz completa y Sherman-Morrison-Woodbury.
7. **QR Dual Metric:** Medición simultánea de $\|Q^T Q - I\|_2$ y $\|A - QR\|_F / \|A\|_F$.
8. **Conditioning Sweep:** Barrido sintético $\kappa(A) \in [1, 10^{16}]$.
9. **Diferencial Cross-Language:** JAX $\leftrightarrow$ C++ $\leftrightarrow$ Rust bit-a-bit con JVP/VJP.
10. **PMTP Fuzzing:** Validación de parser binario, anti-replay, límite 100MB y HMAC.
