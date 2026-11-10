"""Reproduce TODA la práctica de principio a fin con una semilla global.

Uso:
    python experiments/run_all.py                 # todo, semilla 0
    python experiments/run_all.py --seed 42       # otra semilla
    python experiments/run_all.py --omitir complejidad tuning   # saltar pasos lentos

Pasos: (1) dataset sintético + figura, (2) pruebas pytest, (3) experimentos por defecto,
(4) complejidad, (5) ajuste de hiperparámetros.
Nota: las métricas de desempeño son reproducibles con la misma semilla; los TIEMPOS
dependen de la máquina y varían ligeramente entre ejecuciones.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np

from experiments import run_complexity, run_defaults, run_tuning
from src.datasets import make_synthetic, plot_synthetic, save_synthetic_csv


def paso(titulo):
    print(f"\n=== {titulo} ===", flush=True)
    return time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="semilla global")
    ap.add_argument("--omitir", nargs="*", default=[],
                    choices=["tests", "defaults", "complejidad", "tuning"])
    args = ap.parse_args()
    np.random.seed(args.seed)      # por si alguna librería usa el generador global

    t0 = paso(f"1) Dataset sintético (semilla {args.seed})")
    X, y, meta = make_synthetic(seed=args.seed)
    print("  CSV:", save_synthetic_csv(X, y, meta))
    print("  Figura:", plot_synthetic(X, y, meta))

    if "tests" not in args.omitir:
        paso("2) Pruebas (pytest)")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=RAIZ)
        if r.returncode != 0:
            sys.exit("Las pruebas fallaron: se detiene la ejecución.")
    if "defaults" not in args.omitir:
        paso("3) Experimentos con parámetros por defecto")
        run_defaults.ejecutar(seed=args.seed)
    if "complejidad" not in args.omitir:
        paso("4) Análisis de complejidad")
        run_complexity.ejecutar(seed=args.seed)
    if "tuning" not in args.omitir:
        paso("5) Ajuste de hiperparámetros (CV anidada)")
        run_tuning.ejecutar(seed=args.seed)

    print(f"\nListo en {(time.perf_counter() - t0) / 60:.1f} min. Resultados en results/ y data/.")


if __name__ == "__main__":
    main()
