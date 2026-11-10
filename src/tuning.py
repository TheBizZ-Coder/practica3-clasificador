"""Ajuste de hiperparámetros con VALIDACIÓN CRUZADA ANIDADA.

Idea (muy importante para explicarla):
  * CV EXTERNO (10 folds): estima el desempeño "honesto" del procedimiento completo.
  * CV INTERNO (5 folds): dentro del train de cada fold externo, elige los hiperparámetros.
Así el fold de prueba externo NUNCA se usa para elegir parámetros, y la estimación
no sale optimista (cosa que sí pasaría si reportáramos best_score_ del GridSearch).
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, wilcoxon
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     validation_curve)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.classifiers import KNN
from src.models import crear_modelo
from src.validation import crear_particiones

RAIZ = Path(__file__).resolve().parent.parent
TABLAS = RAIZ / "results" / "tables"
FIGURAS = RAIZ / "results" / "figures"

CV_INTERNO = 5
N_ITER_SVM = 40
N_ITER_MLP = 20
MODELOS_AJUSTADOS = ["K-NN", "SVM", "MLP"]


def crear_busqueda(nombre, seed=0, n_jobs=-1):
    """Devuelve el buscador (Grid o Randomized) para un modelo, sobre un Pipeline con escalador.

    - K-NN: rejilla pequeña (44 combinaciones) -> GridSearchCV exhaustivo.
    - SVM y MLP: espacio grande/continuo -> RandomizedSearchCV con distribuciones log-uniformes
      (C, gamma, alpha y learning_rate_init varían en órdenes de magnitud, de ahí la escala log).
    """
    cv = StratifiedKFold(CV_INTERNO, shuffle=True, random_state=seed)
    if nombre == "K-NN":
        pipe = Pipeline([("escalador", StandardScaler()), ("clf", KNN())])
        rejilla = {
            "clf__k": list(range(1, 22, 2)),             # 1,3,5,...,21
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        }
        return GridSearchCV(pipe, rejilla, cv=cv, scoring="accuracy", n_jobs=n_jobs, refit=True)

    if nombre == "SVM":
        # max_iter acota los casos patológicos (p. ej. kernel poly con gamma y C enormes).
        pipe = Pipeline([("escalador", StandardScaler()), ("clf", SVC(max_iter=200_000))])
        espacio = {
            "clf__kernel": ["linear", "rbf", "poly"],
            "clf__C": loguniform(1e-2, 1e3),
            "clf__gamma": loguniform(1e-4, 1e1),          # lo ignora el kernel linear
        }
        return RandomizedSearchCV(pipe, espacio, n_iter=N_ITER_SVM, cv=cv, scoring="accuracy",
                                  n_jobs=n_jobs, random_state=seed, refit=True)

    if nombre == "MLP":
        pipe = Pipeline([("escalador", StandardScaler()),
                         ("clf", MLPClassifier(max_iter=500, random_state=seed))])
        espacio = {
            "clf__hidden_layer_sizes": [(10,), (50,), (100,), (100, 50)],
            "clf__activation": ["relu", "tanh"],
            "clf__alpha": loguniform(1e-5, 1e-1),
            "clf__learning_rate_init": loguniform(1e-4, 1e-1),
            "clf__early_stopping": [True, False],
        }
        return RandomizedSearchCV(pipe, espacio, n_iter=N_ITER_MLP, cv=cv, scoring="accuracy",
                                  n_jobs=n_jobs, random_state=seed, refit=True)
    raise ValueError(nombre)


def _limpiar(params):
    """Quita el prefijo 'clf__' y convierte a tipos nativos para guardar en CSV."""
    return {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


def validacion_anidada(X, y, nombre, dataset, seed=0, n_jobs=-1):
    """CV anidada de un modelo. Devuelve un DataFrame con una fila por fold externo.

    En cada fold externo: (1) la búsqueda elige parámetros SOLO con el train externo (CV interno);
    (2) se evalúa el modelo ajustado y el modelo por defecto sobre el MISMO test externo
    (comparación pareada).
    """
    filas = []
    particiones = crear_particiones(X, y, "kfold", seed)       # mismos folds que en la Fase 4
    for i, (tr, te) in enumerate(particiones):
        t0 = time.perf_counter()
        busqueda = crear_busqueda(nombre, seed, n_jobs).fit(X[tr], y[tr])
        t_busqueda = time.perf_counter() - t0

        acc_ajustado = accuracy_score(y[te], busqueda.predict(X[te]))
        defecto = clone(crear_modelo(nombre, seed)).fit(X[tr], y[tr])
        acc_defecto = accuracy_score(y[te], defecto.predict(X[te]))
        filas.append({
            "dataset": dataset, "modelo": nombre, "fold": i,
            "acc_defecto": acc_defecto, "acc_ajustado": acc_ajustado,
            "acc_cv_interno": busqueda.best_score_, "t_busqueda_s": t_busqueda,
            "mejores_parametros": str(_limpiar(busqueda.best_params_)),
        })
        print(f"    [{dataset} | {nombre}] fold {i + 1}/{len(particiones)}: "
              f"defecto={acc_defecto:.3f} ajustado={acc_ajustado:.3f} ({t_busqueda:.0f}s)", flush=True)
    return pd.DataFrame(filas)


def comparar(df_folds):
    """Tabla 'default vs ajustado' con prueba de Wilcoxon pareada sobre los 10 folds."""
    filas = []
    for (ds, mod), g in df_folds.groupby(["dataset", "modelo"], sort=False):
        d, a = g["acc_defecto"].values, g["acc_ajustado"].values
        dif = a - d
        if np.allclose(dif, 0):
            p = 1.0                      # sin ninguna diferencia, Wilcoxon no está definido
        else:
            p = wilcoxon(a, d, zero_method="wilcox", alternative="two-sided").pvalue
        filas.append({
            "dataset": ds, "modelo": mod,
            "acc_defecto": f"{d.mean():.4f} ± {d.std(ddof=1):.4f}",
            "acc_ajustado": f"{a.mean():.4f} ± {a.std(ddof=1):.4f}",
            "mejora": round(a.mean() - d.mean(), 4),
            "folds_mejor/igual/peor": f"{(dif > 0).sum()}/{(dif == 0).sum()}/{(dif < 0).sum()}",
            "p_wilcoxon": round(float(p), 4),
            "significativa_(p<0.05)": "sí" if p < 0.05 else "no",
        })
    return pd.DataFrame(filas)


def parametros_finales(X, y, nombre, seed=0, n_jobs=-1):
    """Búsqueda sobre TODO el dataset (CV de 5 folds): da los parámetros recomendados.

    Ojo: su puntaje (best_score_) es optimista; el desempeño se estima con la CV anidada.
    """
    b = crear_busqueda(nombre, seed, n_jobs).fit(X, y)
    return {"modelo": nombre, "mejores_parametros": str(_limpiar(b.best_params_)),
            "acc_cv_5_folds_(optimista)": round(b.best_score_, 4)}


def curva_validacion_knn(X, y, dataset, seed=0):
    """Accuracy (train y validación, 10 folds) en función de k para K-NN, por tipo de peso."""
    ks = list(range(1, 22, 2))
    cv = StratifiedKFold(10, shuffle=True, random_state=seed)
    filas = []
    for pesos in ("uniform", "distance"):
        pipe = Pipeline([("escalador", StandardScaler()), ("clf", KNN(weights=pesos))])
        tr, va = validation_curve(pipe, X, y, param_name="clf__k", param_range=ks,
                                  cv=cv, scoring="accuracy", n_jobs=-1)
        for j, k in enumerate(ks):
            filas.append({"dataset": dataset, "weights": pesos, "k": k,
                          "acc_train": tr[j].mean(), "acc_val": va[j].mean(), "acc_val_std": va[j].std(ddof=1)})
    return pd.DataFrame(filas)


def graficar_curvas(df):
    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 4.8))
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        for pesos, c in (("uniform", "#1f77b4"), ("distance", "#d62728")):
            g = df[(df.dataset == ds) & (df.weights == pesos)]
            ax.plot(g.k, g.acc_val, "o-", color=c, label=f"validación ({pesos})")
            ax.fill_between(g.k, g.acc_val - g.acc_val_std, g.acc_val + g.acc_val_std, color=c, alpha=0.12)
            ax.plot(g.k, g.acc_train, "--", color=c, alpha=0.6, label=f"entrenamiento ({pesos})")
        ax.set_title(f"K-NN: accuracy vs k – {ds}")
        ax.set_xlabel("k")
        ax.set_ylabel("accuracy (10-fold)")
        ax.set_xticks(range(1, 22, 2))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / "knn_curva_validacion_k.png", dpi=130)
    plt.close(fig)
