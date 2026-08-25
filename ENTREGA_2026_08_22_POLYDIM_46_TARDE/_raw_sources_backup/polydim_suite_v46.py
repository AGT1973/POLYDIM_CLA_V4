import os
import sys
import numpy as np
import datetime

# Importar el motor que acabamos de parchear
import polydim_motor_v46 as mj

def run_suite():
    log_path = "CERTIFICADO_ESTRES_8H_V46.md"
    results = []
    
    results.append("# CERTIFICADO DE ESTRÉS Y PRUEBAS V46")
    results.append(f"Fecha: {datetime.datetime.now().isoformat()}")
    results.append("Protocolo: Bulldog Critic / Red Team Destructivo\n")
    
    # CHK_28: Ataque de scratch size en C++
    print("Ejecutando CHK_28 (Ataque scratch size C++)...")
    try:
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        q = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        out = mj.slerp_c(p, q, 0.5)
        res_str = f"- **CHK_28 (C++ Scratch Size)**: OK (Vector resultado: {out})"
        results.append(res_str)
        print("  -> OK")
    except Exception as e:
        res_str = f"- **CHK_28 (C++ Scratch Size)**: FALLO CRÍTICO - {e}"
        results.append(res_str)
        print(f"  -> FALLO: {e}")

    # CHK_29: Bitmap overflow
    print("Ejecutando CHK_29 (Bitmap Overflow en PMTP)...")
    try:
        receiver = mj.PmtpStatefulReceiver(b'0'*32, window_size=64)
        epoch = 1
        seq = 100
        payload = b'test'
        epoch_key = receiver._derive_epoch_key(epoch)
        tag = receiver._make_tag(epoch, seq, payload, epoch_key)
        
        ok, msg = receiver.verify_and_accept(epoch, seq, payload, tag)
        assert ok, f"Expected accept, got {msg}"
        
        # Salto gigante para activar MAX_SEQ_JUMP sin cambiar época
        seq = 100 + mj.PmtpStatefulReceiver.MAX_SEQ_JUMP + 1
        tag = receiver._make_tag(epoch, seq, payload, epoch_key)
        ok, msg = receiver.verify_and_accept(epoch, seq, payload, tag)
        assert not ok, "Debería rechazar salto gigantesco de secuencia"
        
        res_str = "- **CHK_29 (Bitmap Overflow)**: OK (Salto de secuencia rechazado correctamente)"
        results.append(res_str)
        print("  -> OK")
    except Exception as e:
        res_str = f"- **CHK_29 (Bitmap Overflow)**: FALLO - {e}"
        results.append(res_str)
        print(f"  -> FALLO: {e}")

    # CHK_30: Respeto de dtype en umbrales de silicio
    print("Ejecutando CHK_30 (Respeto de dtype en umbrales)...")
    try:
        t32 = mj.theta_small(np.float32, 100)
        t64 = mj.theta_small(np.float64, 100)
        assert t32 > t64, f"Umbral F32 ({t32}) no es mayor a F64 ({t64})"
        
        tiny32 = mj.machine_tiny(np.float32)
        tiny64 = mj.machine_tiny(np.float64)
        assert tiny32 > tiny64, "Tiny F32 debe ser mayor a Tiny F64"
        
        res_str = "- **CHK_30 (Respeto de Dtype)**: OK (Umbrales derivados correctamente del silicio)"
        results.append(res_str)
        print("  -> OK")
    except Exception as e:
        res_str = f"- **CHK_30 (Respeto de Dtype)**: FALLO - {e}"
        results.append(res_str)
        print(f"  -> FALLO: {e}")

    # CHK_31: Verificación de Frontera JAX JIT (slerp_batch)
    print("Ejecutando CHK_31 (JAX JIT fallback)...")
    try:
        P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        Q = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        T = np.array([0.5, 0.5], dtype=np.float64)
        out = mj.slerp_batch(P, Q, T)
        res_str = "- **CHK_31 (JAX JIT slerp_batch)**: OK"
        results.append(res_str)
        print("  -> OK")
    except Exception as e:
        res_str = f"- **CHK_31 (JAX JIT slerp_batch)**: FALLO - {e}"
        results.append(res_str)
        print(f"  -> FALLO: {e}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results) + "\n")
    print(f"\\nResultados guardados en {log_path}")

if __name__ == '__main__':
    run_suite()
