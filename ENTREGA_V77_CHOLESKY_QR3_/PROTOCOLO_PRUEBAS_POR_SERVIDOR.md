# 🔬 PROTOCOLO DE BENCHMARKS Y PRUEBAS POR SERVIDOR (POLYDIM V77)
*Guía de Ejecución Práctica según el Hardware Disponible para cada Alumno / Servidor*

---

## 1. Pruebas en Servidor Local / Laptop (Solo CPU x86_64 / ARM / Apple Silicon)

Cualquier alumno puede ejecutar estas pruebas sin necesidad de GPU ni créditos en la nube:

### Test 1.1: Autocompilación Nativa Zero-Trust
- **Objetivo:** Verificar que el monolito extraiga, compile con `cl.exe` (Windows), `g++` (Linux) o `rustc`, y enlace las DLLs/so en caliente.
- **Comando:**
  ```bash
  python polydim_v77_monolito.py
  ```
- **Métrica esperada:** El mensaje *"POLYDIM V75/V77 MONOLITH - Arquitectura Swarm (Epoch/INT8) SOTA Lista"* y la creación atómica de los binarios en `~/.cache/polydim_ffi/`.

### Test 1.2: Auditoría del Escudo Epsilon en Colisión Exacta ($x = y$)
- **Objetivo:** Demostrar que el gradiente en la colisión de dos agentes latentes no explota a `NaN`.
- **Snippet de prueba:**
  ```python
  import jax, jax.numpy as jnp
  from polydim_v77_monolito import GeodesicKernels

  x = jnp.array([1.0, 0.0, 0.0, 0.0])
  y = jnp.array([1.0, 0.0, 0.0, 0.0]) # Colisión 100% idéntica
  grad_fn = jax.grad(lambda a, b: jnp.sum(GeodesicKernels._log_map_unit(a, b)), argnums=0)
  g = grad_fn(x, y)
  print("Gradiente en colisión:", g)
  assert not jnp.isnan(g).any(), "¡FALLO: Gradiente explotó a NaN!"
  print("✅ PASÓ: Cero singularidades.")
  ```

---

## 2. Pruebas en Servidor con GPU (NVIDIA RTX 3080/4090, T4, A100, H100)

Para servidores con soporte CUDA y memoria masiva:

### Test 2.1: Benchmarking de Cholesky-QR3 vs Gram-Schmidt ($D = 10^5 \to 10^6$)
- **Objetivo:** Medir el *speedup* de Cholesky-QR3 frente a `jnp.linalg.qr` al ortogonalizar planos de rotación en alta dimensión.
- **Snippet de prueba:**
  ```python
  import time, jax, jax.numpy as jnp
  from polydim_v77_monolito import CliffordRotors

  D = 100_000
  W = jax.random.normal(jax.random.PRNGKey(42), (D, 2))

  # Medir Cholesky-QR3
  t0 = time.perf_counter()
  Q_chol = CliffordRotors.cholesky_qr3(W).block_until_ready()
  t_chol = time.perf_counter() - t0

  # Medir QR estándar
  t0 = time.perf_counter()
  Q_qr, _ = jnp.linalg.qr(W)
  Q_qr = Q_qr.block_until_ready()
  t_qr = time.perf_counter() - t0

  # Error de ortogonalidad ||Q^T Q - I||
  err_chol = jnp.linalg.norm(Q_chol.T @ Q_chol - jnp.eye(2))
  err_qr = jnp.linalg.norm(Q_qr.T @ Q_qr - jnp.eye(2))

  print(f"Cholesky-QR3: {t_chol*1000:.2f}ms (Error: {err_chol:.2e})")
  print(f"QR Clásico:   {t_qr*1000:.2f}ms (Error: {err_qr:.2e})")
  ```

### Test 2.2: Reducción Jerárquica INT8 (`XLAQuantizer`)
- **Objetivo:** Comprobar que la reducción $O(\log P)$ no satura el bus PCIe al transmitir tensores masivos.

---

## 3. Pruebas en Cloud TPU (Google Colab TPU v2/v3, Kaggle TPU v3-8)

Para entornos con aceleración matricial masiva (Matrix Multiply Units - MXUs):

### Test 3.1: Rendimiento Asintótico en $D = 1,000,000$
- **Objetivo:** Demostrar que los GEMMs de Cholesky-QR3 aprovechan el 100% de las unidades sistólicas de la TPU, donde Gram-Schmidt secuencial causaría estancamiento (*pipeline stall*).
- **Métrica de éxito:** Error de ortogonalidad inferior a $10^{-7}$ con tiempo de cómputo inferior a 50ms en TPU v3.

---

## 4. Pruebas de Red y Enjambre Distribuido (Multi-Agente PMTP)

Para probar la comunicación real entre dos o más procesos/servidores:

### Test 4.1: Enlace TCP y Sincronización de Épocas
- **Objetivo:** Levantar dos agentes (`PMTPAgentBridge`) en puertos distintos y enviar un tensor ND cuantizado.
- **Snippet de prueba:**
  ```python
  import time, jax.numpy as jnp
  from polydim_v77_monolito import PMTPAgentBridge

  # Crear Agente Receptor (A) y Emisor (B)
  agente_A = PMTPAgentBridge(port=9001)
  agente_A.start_server()

  agente_B = PMTPAgentBridge(port=9002)

  # Enviar tensor latente de 10,000D
  tensor_nd = jnp.ones((10000,), dtype=jnp.float32) * 0.5
  ok = agente_B.send_tensor("127.0.0.1", 9001, tensor_nd, receiver_id=agente_A.agent_id)

  time.sleep(0.2)
  sender, recibido = agente_A.inbox.get_nowait()
  print("Transmisión exitosa:", ok)
  print("Tensor recibido (forma):", recibido.shape)
  print("Época sincronizada:", agente_A.epoch_clock.epoch)

  agente_A.stop_server()
  ```
- **Métricas:** Cero pérdida de paquetes, compresión INT8 a 1/4 del ancho de banda FP32, y sincronización causal instantánea en el `EpochClock`.
