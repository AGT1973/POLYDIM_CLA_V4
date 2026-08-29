import time
import os
import sys
import subprocess
import logging

LOG_DIR = r"E:\POLYDIM_EINSOF\SOTA_TEMP"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, "nightly_runner.log"), level=logging.INFO, format='%(asctime)s - %(message)s')

def run_tests():
    logging.info("Iniciando ciclo de stress (pytest) sobre V79 Fixed")
    try:
        env = os.environ.copy()
        env["JAX_ENABLE_X64"] = "1"
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "E:\\POLYDIM_EINSOF\\ENTREGA_V79_BULLDOG_\\test_polydim_v79.py", "-v"],
            env=env,
            capture_output=True,
            text=True,
            timeout=1200
        )
        if result.returncode == 0:
            logging.info("Ciclo OK. Tests pasaron.")
        else:
            logging.error(f"Fallo en ciclo de stress:\n{result.stdout}\n{result.stderr}")
            with open(os.path.join(LOG_DIR, "latest_failure.txt"), "w", encoding="utf-8") as f:
                f.write(result.stdout + "\n" + result.stderr)
    except Exception as e:
        logging.error(f"Error critico en runner: {e}")

if __name__ == "__main__":
    logging.info("Nightly Autonomous Runner Iniciado (D=10^6 ready)")
    iterations = 0
    while True:
        iterations += 1
        logging.info(f"--- Iteracion {iterations} ---")
        run_tests()
        time.sleep(30) # Loop rapido de stress
