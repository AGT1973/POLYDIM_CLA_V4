# 🐕 RED TEAM / BULLDOG CRITIC WHITEBOOK: FRACTURA DEL CHERN NUMBER Y SOLUCIÓN FHS SOTA

## 1. DIAGNÓSTICO DE LA FRACTURA (EL CÓDIGO ACTUAL ES BASURA)

La implementación actual que computa la curvatura de Berry en la red discreta como $F_{12} = \text{Im} \log(U_1 U_2 U_3 U_4)$ a partir de los links no normalizados $U_\mu(k) = \langle \psi(k) | \psi(k+\mu) \rangle$ extraídos directamente de `jnp.linalg.eigh`, padece de **dos fallos críticos y destructivos** que garantizan la generación de ruido topológico puro, especialmente en dimensiones asintóticas ($D = 10^6$).

### A. La Trampa del Gauge Aleatorio y el Branch Cut de la Discontinuidad
El solver `eigh` diagonaliza matrices hermíticas, pero los autovectores propios están definidos salvo una fase arbitraria global. Es decir, en cada punto del espacio recíproco $k$, el solver asigna una fase aleatoria continua $e^{i\theta(k)}$. 
Si denotamos el autovector ideal suave como $|\phi(k)\rangle$, el solver devuelve:
$$|\psi(k)\rangle = e^{i\theta(k)}|\phi(k)\rangle$$

Cuando calculamos el link no normalizado, absorbemos esa fase:
$$U_\mu(k) = e^{-i\theta(k)} e^{i\theta(k+\mu)} \langle \phi(k) | \phi(k+\mu) \rangle$$

El problema asintótico no ocurre si se multiplica *perfectamente* el producto de la plaqueta $U_{12} = U_1 U_2 U_3 U_4$ y *luego* se saca el logaritmo, ya que las fases se cancelan analíticamente. El desastre ocurre en implementaciones flotantes discretas:
1. Si el código actual implementa la suma de fases logarítmicas (típicamente hecho para evitar el underflow en productos encadenados), es decir: $F_{12} \approx \text{Im} \log U_x(k) + \text{Im} \log U_y(k+\hat{x}) - \dots$, el ruido aleatorio $\theta(k)$ inyecta saltos de fase salvajes.
2. Al estar confinados a la rama principal del logaritmo complejo $(-\pi, \pi]$, las fases grandes inducen saltos del branch cut artificiales (vórtices de Dirac fantasmas). Cuando integras sobre toda la zona de Brillouin (BZ), la suma no colapsa a un invariante entero, colapsa a puro ruido termodinámico numérico.

### B. Colapso Subnormal en Alta Dimensión ($D=10^6$)
En $D=10^6$, el producto interno discreto entre dos puntos de red adyacentes no es perfectamente $1$. Hay una pérdida de traslape:
$$|\langle \psi(k) | \psi(k+\mu) \rangle| = 1 - \epsilon$$
Sin normalizar los links (Gauge Fixing a $U(1)$ estricto), el producto multiplicativo de una plaqueta es $\sim (1-\epsilon)^4$. En mallas finas, esto se propaga. Peor aún, si la banda tiene curvatura fuerte, los overlaps disminuyen. Al multiplicar 4 tensores complejos en $D=10^6$ que son $<1$, empujas la mantisa flotante hacia subnormales (underflow). La fase de un subnormal en JAX devuelve `NaN` al invocar `jnp.angle` o `jnp.log`. Todo el tensor muere de manera silenciosa.

---

## 2. LA SOLUCIÓN SOTA: FUKUI-HATSUGAI-SUZUKI (FHS) ESTRICTO

El artículo seminal de *Fukui, Hatsugai, y Suzuki (2005)* demostró que la única manera geométricamente consistente de computar invariantes de Chern en una red discreta sin depender de un gauge global (y por ende ignorando la aleatoriedad de `eigh`) es forzando una conexión $U(1)$ en los links.

**Protocolo Matemático:**
1. **Link U(1) Normalizado:**
   $$U_\mu(k) = \frac{\langle \psi(k) | \psi(k+\mu) \rangle}{\left| \langle \psi(k) | \psi(k+\mu) \rangle \right|}$$
   Esto mapea cada link puramente a la variedad $U(1)$.
2. **Plaqueta de Flujo de Lattice:**
   $$F_{12}(k) = \log\left[ U_x(k) U_y(k+\hat{x}) U_x^{-1}(k+\hat{y}) U_y^{-1}(k) \right]$$
   Donde se escoge la rama principal de $\log(z)$, es decir, $F_{12}(k) = i \arg(U_{12}(k))$, confinada a $(-\pi, \pi]$.
3. **Invariante de Chern Entero:**
   $$C = \frac{1}{2\pi i} \sum_{k \in BZ} F_{12}(k)$$
   El uso estricto del branch principal restringe los monopolos, garantizando que $C$ sea un número entero exacto, inmune a las fases aleatorias iniciales de `eigh`.

---

## 3. IMPLEMENTACIÓN JAX ASINTÓTICA (ZERO-WASTE)

```python
import jax
import jax.numpy as jnp

def fhs_chern_number(psi_grid, eps=1e-15):
    """
    Computa el número de Chern usando el método SOTA de Fukui-Hatsugai-Suzuki.
    
    Args:
        psi_grid: Array JAX de forma (Nx, Ny, D). 
        eps: Epsilon de seguridad para evitar división por cero en degeneración estricta.
        
    Returns:
        chern_int: Invariante de Chern topológico (entero exacto).
    """
    psi_x_shifted = jnp.roll(psi_grid, shift=-1, axis=0)
    psi_y_shifted = jnp.roll(psi_grid, shift=-1, axis=1)
    
    link_x_raw = jnp.sum(jnp.conj(psi_grid) * psi_x_shifted, axis=-1)
    link_y_raw = jnp.sum(jnp.conj(psi_grid) * psi_y_shifted, axis=-1)
    
    # Gauge Fixing Estricto: Forzar los links a la variedad U(1)
    U_x = link_x_raw / (jnp.abs(link_x_raw) + eps)
    U_y = link_y_raw / (jnp.abs(link_y_raw) + eps)
    
    # Flujo en la Plaqueta
    U_x_shifted_y = jnp.roll(U_x, shift=-1, axis=1)
    U_y_shifted_x = jnp.roll(U_y, shift=-1, axis=0)
    
    plaquette_product = U_x * U_y_shifted_x * jnp.conj(U_x_shifted_y) * jnp.conj(U_y)
    
    F_12 = jnp.angle(plaquette_product)
    
    chern_number = jnp.sum(F_12) / (2.0 * jnp.pi)
    
    return jnp.round(chern_number).astype(jnp.int32)
```

**Conclusión del Sabueso:**
Si no implementas esta normalización explícita sobre la variedad $U(1)$, tu código seguirá fallando en alta dimensionalidad.
