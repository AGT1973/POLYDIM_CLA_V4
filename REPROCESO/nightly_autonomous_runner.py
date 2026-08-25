import os
import time
import subprocess
import datetime

LOG_FILE = "E:\\POLYDIM_EINSOF\\REPROCESO\\NOCTURNO_TELEMETRIA_CONTINUA.md"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(f"\n\n# INICIO MODO NOCTURNO V63 ESTOCÁSTICO: {datetime.datetime.now()}\n")

iteration = 0
while True:
    iteration += 1
    t0 = time.time()
    
    # Ejecutamos el fuzzer estocástico de V63
    res = subprocess.run(["python", "E:\\POLYDIM_EINSOF\\REPROCESO\\fuzzer_adversarial_v63.py"], capture_output=True, text=True)
    
    t1 = time.time()
    status = "OK" if res.returncode == 0 else "FAIL"
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"- Iter {iteration:05d} | Estado: {status} | Tiempo: {t1-t0:.2f}s | Hora: {datetime.datetime.now().strftime('%H:%M:%S')}\n")
        if status == "FAIL":
            f.write("```\n")
            f.write(res.stdout + "\n" + res.stderr)
            f.write("```\n")
        else:
            f.write(f"  > Output: {res.stdout.strip()}\n")
    
    # Pausa entre iteraciones aleatorias
    time.sleep(15)
