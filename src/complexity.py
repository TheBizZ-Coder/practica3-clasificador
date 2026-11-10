"""Análisis de complejidad: tabla teórica + experimento empírico (log-log).

Notación
--------
n   : muestras de entrenamiento        m : muestras a predecir
d   : número de rasgos                 c : número de clases
k   : vecinos de K-NN                  n_sv : vectores de soporte de la SVM
W   : número de pesos de la red MLP    T : épocas (iteraciones) del MLP
"""
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from src.datasets import make_synthetic
from src.models import NOMBRES, crear_clasificadores

RAIZ = Path(__file__).resolve().parent.parent
TABLAS = RAIZ / "results" / "tables"
FIGURAS = RAIZ / "results" / "figures"


def tabla_teorica():
    """Complejidad teórica de entrenamiento, predicción y memoria de cada modelo."""
    filas = [
        ("Distancia Mínima", "O(n·d)", "O(m·c·d)", "O(c·d)",
         "Entrenar = calcular c medias; predecir = distancia a c centroides."),
        ("1-NN", "O(n·d) (solo almacena)", "O(m·n·d)", "O(n·d)",
         "Compara cada punto de prueba contra los n de entrenamiento."),
        ("K-NN", "O(n·d) (solo almacena)", "O(m·n·d + m·n·log k)", "O(n·d)",
         "Igual que 1-NN más la selección de los k menores (nuestra versión ordena todo: O(m·n·log n))."),
        ("Naive Bayes", "O(n·d)", "O(m·c·d)", "O(c·d)",
         "Media y varianza por clase y rasgo; predecir evalúa c·d gaussianas."),
        ("SVM (kernel RBF, libsvm)", "entre O(n²·d) y O(n³)", "O(m·n_sv·d)", "O(n_sv·d)",
         "Resuelve un problema cuadrático; la predicción depende de los vectores de soporte."),
        ("MLP", "O(T·n·W)", "O(m·W)", "O(W)",
         "Cada época recorre los n datos con coste proporcional a W (propagación + retropropagación)."),
    ]
    return pd.DataFrame(filas, columns=["Modelo", "Entrenamiento", "Predicción", "Memoria", "Comentario"])


# Exponentes teóricos aproximados de t ∝ (variable)^p, para comparar con la pendiente log-log.
# (d fijo/n fijo según el caso; son aproximaciones: constantes y caché alteran lo observado.)
PENDIENTE_TEORICA = {
    "n": {  # variando n (d fijo, m fijo)
        "Distancia Mínima": {"fit": 1, "predict": 0}, "1-NN": {"fit": 1, "predict": 1},
        "K-NN": {"fit": 1, "predict": 1}, "Naive Bayes": {"fit": 1, "predict": 0},
        "SVM": {"fit": "2 a 3", "predict": 1}, "MLP": {"fit": 1, "predict": 0},
    },
    "d": {  # variando d (n fijo, m fijo)
        "Distancia Mínima": {"fit": 1, "predict": 1}, "1-NN": {"fit": 1, "predict": 1},
        "K-NN": {"fit": 1, "predict": 1}, "Naive Bayes": {"fit": 1, "predict": 1},
        "SVM": {"fit": 1, "predict": 1}, "MLP": {"fit": 1, "predict": 1},
    },
}


def _medir(clf, Xtr, ytr, Xte, repeticiones):
    """Mediana de los tiempos de fit y predict (segundos) sobre varias repeticiones."""
    tf, tp = [], []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        clf.fit(Xtr, ytr)
        tf.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        clf.predict(Xte)
        tp.append(time.perf_counter() - t0)
    return float(np.median(tf)), float(np.median(tp))


def _datos(n, d, m, seed):
    """Sintético con n+m muestras y d rasgos (2 informativos + d-2 de ruido), ya escalado."""
    X, y, _ = make_synthetic(n_samples=n + m, n_ruido=d - 2, n_outliers=max(1, (n + m) // 60), seed=seed)
    X = StandardScaler().fit_transform(X)   # el escalado queda fuera del cronómetro
    return X[:n], y[:n], X[n:n + m]


def experimento_empirico(seed=0, valores_n=(500, 1000, 2000, 5000, 10000, 20000), d_fijo=5,
                         valores_d=(2, 5, 10, 20, 50, 100, 200), n_fijo=2000, m=1000):
    """Mide t_fit y t_predict variando n (d fijo) y variando d (n fijo)."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    filas = []
    for variable, valores in (("n", valores_n), ("d", valores_d)):
        for v in valores:
            n, d = (v, d_fijo) if variable == "n" else (n_fijo, v)
            Xtr, ytr, Xte = _datos(n, d, m, seed)
            reps = 3 if n <= 5000 else 1       # en n grandes basta una repetición (ya son lentos)
            for nombre in NOMBRES:
                clf = crear_clasificadores(seed)[nombre]
                t_fit, t_pred = _medir(clf, Xtr, ytr, Xte, reps)
                filas.append({"variable": variable, "valor": v, "n": n, "d": d, "m": m,
                              "modelo": nombre, "t_fit_s": t_fit, "t_predict_s": t_pred})
            print(f"  complejidad: variando {variable} = {v} listo")
    return pd.DataFrame(filas)


def estimar_pendientes(df):
    """Pendiente de la recta log10(t) = a + p·log10(x) por regresión (np.polyfit).

    Se reportan dos: con todos los puntos y solo con la mitad superior de x
    (donde el coste real domina sobre los costes fijos de arranque).
    """
    filas = []
    for (variable, modelo), g in df.groupby(["variable", "modelo"], sort=False):
        g = g.sort_values("valor")
        x = np.log10(g["valor"].values.astype(float))
        for etapa, col in (("fit", "t_fit_s"), ("predict", "t_predict_s")):
            y = np.log10(np.maximum(g[col].values, 1e-9))
            p_todos = np.polyfit(x, y, 1)[0]
            mitad = len(x) // 2
            p_cola = np.polyfit(x[mitad:], y[mitad:], 1)[0]
            filas.append({"variable": variable, "modelo": modelo, "etapa": etapa,
                          "pendiente_todos": round(p_todos, 3), "pendiente_mitad_superior": round(p_cola, 3),
                          "pendiente_teorica": PENDIENTE_TEORICA[variable][modelo][etapa]})
    return pd.DataFrame(filas)


def graficar(df, pend):
    """Gráficas log-log (una figura por variable, paneles fit y predict) con rectas ajustadas."""
    colores = plt.cm.tab10(np.arange(6))
    for variable, etiqueta in (("n", "n (muestras de entrenamiento)"), ("d", "d (rasgos)")):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax, etapa, col in zip(axes, ("fit", "predict"), ("t_fit_s", "t_predict_s")):
            for c, nombre in zip(colores, NOMBRES):
                g = df[(df.variable == variable) & (df.modelo == nombre)].sort_values("valor")
                p = pend[(pend.variable == variable) & (pend.modelo == nombre) & (pend.etapa == etapa)]
                p = p["pendiente_todos"].iloc[0]
                x, y = g["valor"].values.astype(float), np.maximum(g[col].values, 1e-9)
                ax.loglog(x, y, "o-", color=c, label=f"{nombre} (pend. {p:.2f})")
                # recta de regresión log-log
                a, b = np.polyfit(np.log10(x), np.log10(y), 1)[::-1]
                ax.loglog(x, 10 ** (a + b * np.log10(x)), ":", color=c, alpha=0.7)
            ax.set_xlabel(etiqueta)
            ax.set_ylabel(f"tiempo de {etapa} (s)")
            ax.set_title(f"Tiempo de {etapa} vs {variable}")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8)
        fig.tight_layout()
        FIGURAS.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURAS / f"complejidad_vs_{variable}.png", dpi=130)
        plt.close(fig)
