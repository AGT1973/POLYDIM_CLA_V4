import time
import os
import sys
import datetime
import traceback

LOG_FILE = "NOCTURNO_TELEMETRIA_CONTINUA.md"

def write_telemetry(msg):
    timestamp = datetime.datetime.now().isoformat()
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"- [{timestamp}] {msg}\n")
    except Exception as e:
        print(f"Error escribiendo telemetria: {e}", file=sys.stderr)

if __name__ == '__main__':
    write_telemetry("🚀 INICIO MODO NOCTURNO V66: Runner de Silicio Autónomo Activado.")
    iteration = 0
    while True:
        iteration += 1
        try:
            monolith_path = os.path.join("ENTREGA_20260825_", "polydim_v66_monolito.py")
            if os.path.exists(monolith_path):
                write_telemetry(f"Iteración #{iteration}: Ejecutando verificación de silicio V66 sobre {monolith_path}...")
                ret = os.system(f'python "{monolith_path}" > nul 2>&1')
                if ret == 0:
                    write_telemetry(f"Iteración #{iteration}: ✅ Verificación de monolito V66 completada PASS.")
                else:
                    write_telemetry(f"Iteración #{iteration}: ⚠️ Verificación de monolito V66 retorno código {ret}.")
            else:
                write_telemetry(f"Iteración #{iteration}: Monolito V66 no encontrado en {monolith_path}.")
        except Exception as err:
            write_telemetry(f"Iteración #{iteration}: 💥 Excepción en runner V66: {err}\n{traceback.format_exc()}")
        
        time.sleep(300)
