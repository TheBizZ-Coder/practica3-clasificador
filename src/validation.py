"""Esquemas de validación y métricas.

- Hold-Out 70/30 estratificado repetido con 10 semillas.
- K-Fold estratificado con k=10.

Las particiones se generan UNA sola vez (``crear_particiones``) y se reutilizan
para todos los modelos, de modo que la comparación sea justa (mismos datos de
train/test para todos).
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

N_SEMILLAS_HOLDOUT = 10
K_FOLDS = 10


def crear_particiones(X, y, esquema, seed=0):
    """Lista de (idx_train, idx_test).

    esquema='holdout': 10 particiones 70/30 estratificadas, con semillas seed..seed+9.
    esquema='kfold'  : 10 pliegues estratificados (barajados con la semilla).
    """
    if esquema == "holdout":
        idx = np.arange(len(y))
        partes = []
        for s in range(seed, seed + N_SEMILLAS_HOLDOUT):
            tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=s)
            partes.append((tr, te))
        return partes
    if esquema == "kfold":
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
        return list(skf.split(X, y))
    raise ValueError("esquema debe ser 'holdout' o 'kfold'")


def evaluar(modelo, X, y, particiones, clases):
    """Evalúa un modelo en todas las particiones.

    Devuelve
    --------
    df : DataFrame con una fila por partición (accuracy, f1_macro, t_fit, t_predict).
    cm : matriz de confusión AGREGADA (suma de las de todas las particiones).
    """
    filas = []
    cm = np.zeros((len(clases), len(clases)), dtype=int)
    for i, (tr, te) in enumerate(particiones):
        m = clone(modelo)                       # copia limpia: nada se arrastra entre particiones

        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])                     # el Pipeline ajusta el escalador solo con train
        t_fit = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = m.predict(X[te])
        t_pred = time.perf_counter() - t0

        filas.append({
            "particion": i,
            "accuracy": accuracy_score(y[te], pred),
            "f1_macro": f1_score(y[te], pred, average="macro"),
            "t_fit_s": t_fit,
            "t_predict_s": t_pred,
        })
        cm += confusion_matrix(y[te], pred, labels=clases)
    return pd.DataFrame(filas), cm


def resumir(df):
    """Media y desviación estándar de cada métrica sobre las particiones."""
    cols = ["accuracy", "f1_macro", "t_fit_s", "t_predict_s"]
    out = {}
    for c in cols:
        out[f"{c}_media"] = df[c].mean()
        out[f"{c}_std"] = df[c].std(ddof=1)
    return out
