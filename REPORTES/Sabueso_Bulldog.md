# Reporte Red Team - Revisión Arquitectónica de M-of-N y rotate_committee
**Analista**: Sabueso Bulldog

## 1. Análisis de Vulnerabilidades y Edge-Cases

### 1.1. Condiciones de Carrera en la Transición (`rotate_committee`)
El "baile sutil" entre el estado del comité viejo y el nuevo al emitir ledgers (`_issue`) presenta un vector crítico. Dado que la recolección de firmas es asincrónica:
*   **Emisión intercalada (Interleaved Issuance):** Si `rotate_committee` acumula firmas y muta el estado parcialmente, otro hilo concurrente llamando a `_issue` podría capturar un estado de comité inconsistente (ej. mezclando firmas admitidas por la política vieja y la nueva).
*   **Pérdida de Liveness por Deadlock:** Si la función de rotación bloquea el estado global `_com_state` esperando el quórum de firmas externas, bloquea la validación y emisión de todos los ledgers concurrentes.

### 1.2. Fallas del Quórum Asincrónico
*   **Firmas Húerfanas (Orphan Signatures):** En un modelo asincrónico, las firmas para la transición de comité pueden llegar después de que la rotación ya fue abortada. Si no hay un identificador de época (`epoch`) estricto, firmas viejas podrían ser inyectadas maliciosamente para forzar una rotación no deseada (Replay Attack de estados transicionales).
*   **Desgaste de Quórum:** Si un miembro del comité viejo es revocado o su red colapsa durante la recolección de firmas para rotar, el proceso de rotación queda huérfano de por vida si no se implementa un TTL explícito.

### 1.3. Split-View y Bifurcación Catastrófica (Falla de Sincronización)
El diseño asume que si `k > n/2`, se previenen dos subgrupos disjuntos. Sin embargo:
*   **Doble Firma Bizantina:** Un solo nodo malicioso puede violar el protocolo y firmar dos ramas distintas de la historia (ej. ledger N+1a y N+1b), otorgando a ambas el quórum válido `k`.
*   **Rotación en Split-View (Chain Split):** Si se produce un split-view y una rama ejecuta `rotate_committee`, las ramas divergirán en su composición de autoridad. La alarma `serial_conflicts` solo se disparará si los nodos honestos ven ambas ramas (difícil bajo un Eclipse Attack). Una vez que los comités divergen, el sistema colapsará en producción al ser incapaz de reconciliar matemáticamente ambas ramas.

---

## 2. Propuesta de Código Endurecido (Anti-Happy-Path / Anti-Hardcoding)

Para evitar que `_com_state` y `_issue` colisionen, el diseño se refactoriza a un modelo de transición de dos fases (Prepare / Commit) protegido por un cerrojo (lock) de corto alcance y control de monotonía de Épocas (`epoch`).

```python
import threading
import time
from typing import Set, Dict, Tuple, Optional
import logging

class CommitteeState:
    def __init__(self, members: Set[str], threshold: int, epoch: int):
        # Anti-Happy-Path: Obligamos el límite matemáticamente, sin depender del runbook.
        if threshold <= len(members) // 2:
            raise ValueError(f"Threshold {threshold} MUST be > n/2 to prevent disjoint quorums")
        
        # Estado Inmutable para prevenir carreras de datos sucias durante _issue
        self.members = frozenset(members)
        self.threshold = threshold
        self.epoch = epoch

class HardenedCommitteeAuthority:
    def __init__(self, initial_members: Set[str], threshold: int):
        self._lock = threading.RLock()
        self._active_state = CommitteeState(initial_members, threshold, epoch=0)
        self._pending_rotation: Optional[Tuple[CommitteeState, Dict[str, bytes]]] = None
        self._rotation_timeout = 0.0
        
    def request_rotation(self, new_members: Set[str], new_threshold: int, timeout_sec: float = 60.0):
        """Inicia el quórum asincrónico para rotar el comité."""
        with self._lock:
            if self._pending_rotation is not None:
                if time.time() < self._rotation_timeout:
                    raise RuntimeError("Rotation already in progress")
                logging.warning("Previous rotation timed out. Overwriting.")
                
            # Se incrementa el epoch estrictamente
            new_state = CommitteeState(new_members, new_threshold, self._active_state.epoch + 1)
            self._pending_rotation = (new_state, {})
            self._rotation_timeout = time.time() + timeout_sec
            
    def submit_rotation_signature(self, signer_id: str, signature: bytes):
        """Recepción asincrónica de firmas del comité VIEJO aprobando al NUEVO."""
        with self._lock:
            if self._pending_rotation is None or time.time() > self._rotation_timeout:
                self._pending_rotation = None
                raise TimeoutError("No active rotation or rotation timed out")
                
            if signer_id not in self._active_state.members:
                raise ValueError(f"Signer {signer_id} not in ACTIVE committee")
                
            new_state, sigs = self._pending_rotation
            
            # Simulación de verificación criptográfica vinculada a la nueva estructura.
            # verify_signature(signer_id, serialize_state(new_state), signature)
            
            sigs[signer_id] = signature
            
            # COMMIT phase: Si alcanzamos el quórum del VIEJO comité, rotamos atómicamente
            if len(sigs) >= self._active_state.threshold:
                logging.info(f"Quorum reached. Rotating committee to epoch {new_state.epoch}")
                self._active_state = new_state
                self._pending_rotation = None

    def issue_ledger(self, payload: bytes, signatures: Dict[str, bytes]) -> bytes:
        """
        Emisión protegida. 
        Previene usar el comité viejo parcial si ocurre un rotador concurrente.
        """
        with self._lock:
            # Snapshot de estado: inmutable, a prueba de carreras con rotate_committee
            current_state = self._active_state
            
        valid_sigs = 0
        # Fuera del lock para no obstruir el sistema durante cálculos costosos
        for signer, sig in signatures.items():
            if signer in current_state.members:
                # verify_signature(...)
                valid_sigs += 1
                
        if valid_sigs < current_state.threshold:
            raise ValueError(
                f"Insufficient valid signatures: got {valid_sigs}, "
                f"require {current_state.threshold} (Epoch {current_state.epoch})"
            )
            
        # El ledger DEBE incorporar el epoch para rechazar splits cross-epoch.
        return b"ISSUED_" + payload + b"_EPOCH_" + str(current_state.epoch).encode()
```
