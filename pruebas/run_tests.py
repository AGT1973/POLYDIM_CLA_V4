import os
import sys

# Agregar la carpeta de entrega al path de Python para importar el monolito de producción
DELIVERY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ENTREGA_20260827_V73_"))
if DELIVERY_DIR not in sys.path:
    sys.path.insert(0, DELIVERY_DIR)

try:
    from polydim_v73_monolito import run_self_verification
except ImportError as e:
    print(f"Error al importar el monolito desde {DELIVERY_DIR}: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("Iniciando Suite de Pruebas de POLYDIM...")
    try:
        run_self_verification()
    except KeyboardInterrupt:
        print("\nPruebas interrumpidas por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\nFallo crítico durante la ejecución de las pruebas: {e}")
        sys.exit(1)
