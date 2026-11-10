"""Fase 6: ajuste de hiperparámetros (K-NN, SVM, MLP) con CV anidada (5 interno / 10 externo).

Genera en results/tables: tuning_folds.csv, tuning_default_vs_ajustado.csv/.md,
tuning_parametros_finales.csv/.md, tuning_curva_knn.csv; y la figura de la curva de k.
"""
import os
import sys
import warnings
from pathlib import Path

# Silencia ConvergenceWarning también en los procesos de joblib (hereda la variable).
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from experiments.run_defaults import cargar_datasets, guardar_tabla
from src.tuning import (MODELOS_AJUSTADOS, TABLAS, comparar, curva_validacion_knn, graficar_curvas,
                        parametros_finales, validacion_anidada)


def ejecutar(seed=0):
    datasets = cargar_datasets(seed)
    TABLAS.mkdir(parents=True, exist_ok=True)

    # 1) Curva de validación de K-NN (accuracy vs k)
    curvas = pd.concat([curva_validacion_knn(X, y, ds, seed) for ds, (X, y) in datasets.items()])
    curvas.to_csv(TABLAS / "tuning_curva_knn.csv", index=False)
    graficar_curvas(curvas)
    print("  curva de validación de K-NN lista", flush=True)

    # 2) CV anidada por modelo y dataset
    folds = []
    for ds, (X, y) in datasets.items():
        for nombre in MODELOS_AJUSTADOS:
            folds.append(validacion_anidada(X, y, nombre, ds, seed))
    folds = pd.concat(folds)
    folds.to_csv(TABLAS / "tuning_folds.csv", index=False)
    tabla = comparar(folds)
    guardar_tabla(tabla, "tuning_default_vs_ajustado")

    # 3) Parámetros recomendados (búsqueda sobre todo el dataset)
    finales = []
    for ds, (X, y) in datasets.items():
        for nombre in MODELOS_AJUSTADOS:
            finales.append({"dataset": ds, **parametros_finales(X, y, nombre, seed)})
    guardar_tabla(pd.DataFrame(finales), "tuning_parametros_finales")

    print(tabla.to_string(index=False))
    return tabla


if __name__ == "__main__":
    ejecutar(seed=0)
