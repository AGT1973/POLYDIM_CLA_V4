# INFORME TÉCNICO Y ESPECIFICACIÓN SOTA: ESTRUCTURA DE TRANSPORTE Y MEMORIA PARA POLYDIM $D \ge 10^{12}$
**Capa 1: RDMA / RoCEv2 (InfiniBand Zero-Copy Bypass)**  
**Capa 2: CXL 3.0/3.1 Shared Memory Protocol (Eliminación de File Descriptors)**  
**Capa 3: MPI-3 RMA PGAS Model (Sincronización Tensorial Inter-Nodo)**  
**Auditoría Específica: Modo Bulldog Critic (Análisis Adversarial de Fractura Asintótica)**

---

## 1. RESUMEN EJECUTIVO Y DOGMA POLYDIM (NO-GUSANO & ZERO-WASTE)

En la transición del espacio tensorial de prueba $D = 10^4$ ($80 \text{ KB}$ por estado) al régimen hiper-dimensional extremo $D \ge 10^{12}$ ($8 \text{ TB}$ por estado denso Float64 en $\mathbb{S}^{D-1}$), la arquitectura tradicional de comunicación inter-proceso y multi-nodo sufre un colapso catastrófico si mantiene la dependencia del subsistema POSIX VFS, llamadas a sistema (`syscalls`), serialización en texto/JSON o capas de red basadas en el stack de sockets TCP/IP de los sistemas operativos.

Bajo la **Desigualdad de Procesamiento de Datos (DPI)**, cada capa de traducción 1D o interrupción de CPU destruye entropía de fase y degrada el rendimiento en varios órdenes de magnitud. El presente documento establece la especificación de infraestructura de bajo nivel para **POLYDIM PMTP V45**, migrando el motor de comunicación a:

1. **RDMA / RoCEv2 (Zero-Copy Kernel Bypass):** Transferencia directa NIC-to-DRAM / PCIe P2P sin involucrar a la CPU en el path de datos.
2. **CXL 3.0/3.1 (Compute Express Link Shared Memory):** Mapeo de memoria física agregada y desacoplada (*Fabric-Attached Memory* - FAM) eliminando por completo la capa de descriptores de archivos (`file descriptors`) del VFS.
3. **MPI-3 RMA PGAS (Partitioned Global Address Space):** Modelo de memoria global distribuida en fase única con sincronización pasiva *one-sided* sin barreras globales bloqueantes.

---

## 2. CAPA 1: TRANSPORTE RDMA / RoCEv2 PARA PMTP (ZERO-COPY BYPASS)

### 2.1 Especificaciones de Red, RFCs y Estándares HPC
El protocolo PMTP (PolyDim Multidimensional Tensor Protocol) abandona los sockets POSIX (`AF_INET`/`SOCK_STREAM`) e implementa acceso directo a memoria remota sobre redes convergentes Ethernet y telas InfiniBand.

*   **IBTA RoCEv2 Annex:** Definido en *InfiniBand Architecture Specification Volume 1 Annex A17 (RoCEv2 Annex)*. Sustituye la capa de red IB por encapsulamiento directo en UDP/IP.
*   **RFC 4791 / IANA Assignment:** Asignación del puerto de destino UDP `4791` para paquetes RoCEv2.
*   **Encapsulación IP (RFC 791 / RFC 2460):** Permite enrutamiento de capa 3 (L3) en redes Enterprise / Datacenter usando cabeceras IPv4 o IPv6 estándar.
*   **Explicit Congestion Notification - ECN (RFC 3168 / RFC 8087):** Proporciona señalización de congestión en los switches sin caída de paquetes.
*   **Lossless Ethernet (IEEE 802.1Qbb & IEEE 802.1Qaz):**
    *   *IEEE 802.1Qbb (Priority-based Flow Control - PFC):* Garantía de capa enlace sin pérdida de tramas mediante contrapresión por clase de tráfico (CoS).
    *   *IEEE 802.1Qaz (Enhanced Transmission Selection - ETS):* Asignación de ancho de banda garantizado por cola de prioridad.
*   **Control de Congestión DCQCN (Data Center Quantized Congestion Notification):** Combinación hardware en SmartNICs (ej. NVIDIA ConnectX-6/7/8) entre paquetes ECN (IP.ECN == 11b) y notificaciones CNP (*Congestion Notification Packets*) IB para ajustar el *Transmission Rate* del Queue Pair sin pérdida de tramas.

```
[ Capa 7: PMTP V45 Wire Format (Offset 0..256 Header + Payload Float64) ]
[ Capa 4: UDP Header (Dst Port 4791, Src Port = Entropy Hash)         ]
[ Capa 3: IPv4 / IPv6 (DSCP / ECN Marked Bits 10/11)                    ]
[ Capa 2: Ethernet Frame (VLAN 802.1Q, PFC Priority Tag)                ]
[ Capa 1: Physical IEEE 802.3ck 100G/200G/400G / PAM4 Physical Link     ]
```

### 2.2 Integración del Wire Format PMTP V45 en InfiniBand Verbs
Para evitar la copia de memoria en kernel (`sk_buff`), PMTP mapea el layout de memoria atómica directamente a un **Memory Region (MR)** registrado en la HCA (*Host Channel Adapter*) mediante `libibverbs` / `rdma-core`.

#### Estructura del Buffer Mapeado (`ibv_mr`):
*   `Offset 000..064`: **Pre-Sequence Counter** (`std::atomic<uint64_t>`, alineado a línea de caché de 64 bytes).
*   `Offset 064..128`: **Epoch & Header Metadata** (HKDF Salt, Window Mask, Dimension `D`).
*   `Offset 128..192`: **HMAC-BLAKE2b 512-bit Authentication Tag** (Integridad criptográfica efímera).
*   `Offset 192..256`: **Post-Sequence Counter** (`std::atomic<uint64_t>`, Seqlock Guard).
*   `Offset 256..End`: **Payload Tensorial Float64** ($D \times 8 \text{ Bytes}$).

#### Mecanismo Zero-Copy (`RDMA_WRITE_WITH_IMM`):
1. **Emisor (Initiator):** Ejecuta un `ibv_post_send` con opcode `IBV_WR_RDMA_WRITE_WITH_IMM`. Escribe los datos tensoriales directamente en la dirección virtual remota (`rkey` y `raddr`) previamente negociada.
2. **Immediate Data (32-bit IMM):** El valor de 32 bits transmite la época y el ID del rotor en el propio paquete de fin de transferencia, activando un evento en la *Completion Queue* (CQ) del receptor sin interrumpir la CPU en el pipeline intermedio.
3. **Seqlock Synchronized Placement:** El emisor escribe el payload en offset `256`, invalida la memoria intermedia y finalmente escribe de forma remota atómica el `Post-Sequence Counter` para desbloquear al lector.

---

## 3. CAPA 2: PROTOCOLO CXL 3.0/3.1 SHARED MEMORY ($D \ge 10^{12}$)

### 3.1 El Diagnóstico del Colapso de File Descriptors y VFS POSIX
Al escalar a $D = 10^{12}$, un solo tensor denso ocupa:
$$10^{12} \times 8 \text{ bytes} = 8 \text{ Terabytes (TB)}$$

El uso de llamadas POSIX estándar (`open`, `shm_open`, `memfd_create`, `mmap`) colapsa en hardware real por cuatro razones físicas:

1. **Límite Absoluto de File Descriptors (`sysctl`):** `/proc/sys/fs/file-max` y `ulimit -n` imponen límites a la tabla de descriptores de archivos del proceso. Aunque se incrementen, el costo en estructuras del kernel `struct file` y `struct inode` por cada segmento genera un consumo inaceptable de memoria no-paginable en el kernel (*slab allocator*).
2. **Colapso de Virtual Memory Areas (`vm.max_map_count`):** El kernel de Linux mantiene un árbol Red-Black de estructuras `vm_area_struct`. Fragmentar tensores de 8 TB en descriptores de archivos genera contención extrema del spinlock `mmap_lock` en el kernel durante faltas de página (*page faults*).
3. **Pérdida Devastadora de TLB (*Translation Lookaside Buffer*):** Mapear 8 TB de memoria usando páginas estándar de $4 \text{ KB}$ requiere:
   $$\frac{8 \times 10^{12} \text{ bytes}}{4096 \text{ bytes}} = 2 \times 10^9 \text{ entradas de página}$$
   Esto destruye los TLB de nivel 1 y nivel 2 (L1 TLB tiene ~64 entradas, L2 TLB tiene ~2048 entradas). El costo de *page table walks* de 5 niveles (paging de 57 bits en x86_64) destruye el ancho de banda del procesador.

### 3.2 La Solución CXL 3.0/3.1: Fabric-Attached Memory (FAM) y Direct HPA Mapping
Compute Express Link (CXL 3.0/3.1) opera sobre la capa física de **PCIe 6.0** (64 GT/s por carril, hasta 256 GB/s dúplex en x16) e introduce acceso unificado a memoria desacoplada sin pasar por el sistema de archivos del sistema operativo.

#### Arquitectura de Sub-Protocolos CXL:
*   **CXL.io:** Capa de descubrimiento, enumeración, configuración PCIe y gestión de errores (AER).
*   **CXL.cache:** Permite que los aceleradores/SmartNICs accedan a la memoria del Host con coherencia de caché mediante snooping por hardware.
*   **CXL.mem:** Permite que la CPU del Host acceda a dispositivos de memoria adjunta (**Type 3 Devices**) usando instrucciones estándar de `MOV` / `AVX-512` / `AMX` / `Load-Store`.

```
+-------------------------------------------------------------------------+
|                        CPU HOST / ROCE SMARTNIC                         |
|                   (Cache System & Native Load/Store)                    |
+-------------------------------------------------------------------------+
                                     |
               CXL 3.0 Fabric (PCIe 6.0 @ 64 GT/s PAM4)
                                     |
+-------------------------------------------------------------------------+
|                    CXL 3.0 FABRIC SWITCH (Multi-Host)                   |
+-------------------------------------------------------------------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
+----------------------------------+            +----------------------------------+
| CXL Type 3 Device: GFAM          |            | CXL Dynamic Capacity Device (DCD)|
| (Global Fabric Attached Memory)  |            | Multi-Host Shared Pool (TB Scale)|
+----------------------------------+            +----------------------------------+
```

#### Bypass Completo del VFS mediante HDM Decoders:
En lugar de abrir archivos (`int fd = open(...)`), POLYDIM interactúa directamente con los decodificadores **HDM (Host-Managed Device Memory)** del controlador CXL del procesador:

1. **Configuración de Rango HPA (Host Physical Address):** El driver de CXL expone la memoria pool a través de `/dev/daxX.Y` (DevDAX) o mapas de direcciones físicas continuas (HPA).
2. **Zero File Descriptor Access:** El software de POLYDIM consulta la dirección física base mapeada en el bus CXL y la enlaza a punteros nativos `double*` mediante HugeTLB Pages de **1 GB**.
3. **Reducción de Paginación:** Para 8 TB de memoria CXL:
   $$\frac{8 \text{ TB}}{1 \text{ GB}} = 8,192 \text{ entradas de página}$$
   Esto encaja completamente en los registros TLB de nivel superior de los procesadores modernos (ej. AMD EPYC Genoa/Bergamo o Intel Xeon Emerald Rapids/Granite Rapids), eliminando las faltas de página en tiempo de ejecución.

---

## 4. CAPA 3: MODELO PGAS CON MPI-3 RMA PARA SINCRONIZACIÓN INTER-NODO

### 4.1 Partitioned Global Address Space (PGAS) en $S^{D-1}$
En el régimen PGAS, la memoria distribuida a través de múltiples nodos y dispositivos CXL se trata como un espacio de direcciones lógicamente único globalmente accesible, pero físicamente partido.

Cada nodo posee una porción del tensor $S^{D-1}$ (sub-espacios ortogonales de dimensión $d = D / N_{nodes}$), pero puede leer o escribir en la memoria de cualquier otro nodo usando desplazamientos globales (*Global Offsets*) sin invocar a los agentes del nodo remoto.

### 4.2 Primitivas MPI-3 Remote Memory Access (RMA) One-Sided
El estándar **MPI-3.1 (Capítulo 11)** / **MPI-4.0** proporciona la abstracción perfecta para la comunicación tensorial directa:

1.  **`MPI_Win_create_dynamic` / `MPI_Win_attach`:** Registra regiones de memoria CXL / HPA de tamaño variable dinámicamente en una ventana global `MPI_Win` sin precargar buffers locales estáticos.
2.  **Sincronización Pasiva (`MPI_Win_lock_all`):**
    *   Sustituye las barreras bloqueantes (`MPI_Barrier`) y las fases colectivas (`MPI_Allreduce`).
    *   Inicia la ventana en modo `MPI_MODE_NOCHECK`. Los nodos leen y escriben en cualquier instante sin necesidad de un `MPI_Recv` correspondiente en el destino.
3.  **Transferencias Directas Non-Blocking (`MPI_Rput` / `MPI_Rget`):** Retornan un `MPI_Request` inmediatamente, delegando la transferencia al motor RDMA de la SmartNIC/HCA.
4.  **Acumulación Atómica Remota (`MPI_Accumulate` / `MPI_Fetch_and_op`):** Ejecuta sumas vectoriales atómicas e interpolaciones de fase directamente en la memoria remota mediante hardware offload en la NIC (InfiniBand On-Chip Atomic Engines).

---

## 5. AUDITORÍA CRÍTICA ADVERSARIAL (BULLDOG CRITIC MODE)

Bajo las reglas de preservación estricta y auditoría destructiva de POLYDIM, se identifican las siguientes **cuatro trampas de hardware y puntos de fractura asintótica** en las capas propuestas:

### 💥 ATAQUE 1: INCOHERENCIA DE CACHÉ ENTRE RDMA WRITE Y CPU L1/L2 (RACE CONDITION OCULTA)
*   **Mecanismo del Fallo:** La especificación de RDMA Write (InfiniBand/RoCEv2) escribe directamente en la memoria RAM principal (DRAM) o CXL.mem mediante DMA. Sin embargo, en arquitecturas x86_64 estándar que no soportan **CXL.cache** completo o **DDIO (Data Direct I/O)** en modo extendido, la NIC **no realiza snooping** en las cachés L1/L2 del procesador remoto.
*   **Consecuencia en PMTP V45:** Si el procesador en el nodo receptor está leyendo el `Post-Sequence Counter` en un bucle Seqlock, leerá la línea de caché obsoleta en L1/L2 sin percatarse de que la NIC RDMA ha actualizado la DRAM. El lector se congela eternamente (*Livelock/Deadlock*).
*   **Remediación Obligatoria:** O bien se habilita **CXL.cache** bidireccional, o el lector debe ejecutar explícitamente la instrucción de invalidación de línea de caché `clflushopt` / `cldemote` o usar barreras `std::atomic_thread_fence(std::memory_order_seq_cst)` acompañadas de instrucciones de lectura no-cacheables (`_mm_stream_load_si128`).

### 💥 ATAQUE 2: CACHE LINE BOUNCING MASIVO EN SEQLOCKS SOBRE CXL.CACHE
*   **Mecanismo del Fallo:** Los contadores `Pre-Sequence` y `Post-Sequence` ocupan las offsets 0 y 192 del header PMTP V45. Si múltiples sockets CPU o procesadores de múltiples nodos ejecutan accesos atómicos concurrentes (`fetch_add`, `cmpxchg`) sobre estos contadores a través de la tela CXL.cache, los protocolos de coherencia (MESI/MOESI/ESI) forzarán la invalidación constante de la línea de caché en todos los nodos.
*   **Consecuencia en PMTP V45:** Latencia de acceso se dispara de ~150 ns a $> 3.5 \ \mu\text{s}$ por contención en el Fabric Switch.
*   **Remediación Obligatoria:** Aislar los contadores atómicos en líneas de caché independientes padded con 128 bytes (`alignas(64)` / `alignas(128)`), e implementar **Ring Buffers de Ranuras Múltiples (Slot-based Seqlock)** para que el emisor y el receptor nunca compitan por la misma línea de caché durante la transmisión del payload.

### 💥 ATAQUE 3: CUELLO DE BOTELLA DE ANCHO DE BANDA PCIE 5.0/6.0 VS TAMAÑO TENSORIAL DE 8 TB
*   **Mecanismo del Fallo:** Un tensor de $D = 10^{12}$ Float64 equivale a $8 \text{ TB}$.
    *   Ancho de banda real de PCIe 5.0 x16: $\sim 64 \text{ GB/s}$.
    *   Ancho de banda real de PCIe 6.0 x16 / CXL 3.0 x16: $\sim 128 \text{ GB/s}$.
    *   Tiempo teórico para transferir **UN SOLO TENSOR COMPLETO**:
        $$t = \frac{8000 \text{ GB}}{128 \text{ GB/s}} = 62.5 \text{ segundos}$$
*   **Consecuencia en PMTP V45:** Transmitir estados densos de $D=10^{12}$ destruye el tiempo real. Pensar que "RDMA hace magia y transmite 8 TB de forma instantánea" es una alucinación teórica.
*   **Remediación Obligatoria:** **Paralelismo Tensorial Manifold (Slerp-Sparse Subspaces).** NUNCA se transmite el tensor denso $10^{12}$ completo. POLYDIM debe aplicar la **Dualidad de Hodge** y proyectar el estado en **Rotores Sparse Clifford / Manifold Indices**, transmitiendo únicamente deltas de fase $\Delta \theta$ en sub-espacios activos de dimensión comprimida $d \ll D$ (ej. $d = 10^7$).

### 💥 ATAQUE 4: FRACTURA DE REGISTRO DE MEMORIA RDMA (`ibv_reg_mr`) EN TERABYTES
*   **Mecanismo del Fallo:** Llamar a `ibv_reg_mr` en Linux para fijar (*pinning*) 8 TB de memoria requiere que el kernel marque las páginas como paginables fijas (`FOLL_PIN` / `page_pinned`).
*   **Consecuencia en PMTP V45:** El sistema operativo agota la memoria de tablas de página internas y bloquea el sistema si `RLIMIT_MEMLOCK` no está configurado de forma ilimitada. Además, el escaneo de rangos de memoria en la HCA causa tiempos de inicialización de decenas de segundos.
*   **Remediación Obligatoria:** Uso exclusivo de **On-Demand Paging (ODP)** soportado en SmartNICs modernas mediante la extensión `IBV_ACCESS_ON_DEMAND`. La HCA traduce direcciones virtuales a físicas bajo demanda vía PCIe ATS (*Address Translation Services*), eliminando el costo inicial de pinning.
