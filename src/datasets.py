"""Datasets de la práctica.

1. Sintético con RUIDO ESTRUCTURADO (``make_synthetic``).
2. Benchmark Wine (``load_wine_data``).

"Ruido estructurado" significa que el ruido NO es uniforme: está concentrado
donde más daño hace (fronteras entre clases), varía por clase (heterocedástico),
añade rasgos inútiles y un cúmulo de outliers.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

# Nombres de columnas del dataset sintético
COLS_INFORMATIVAS = ["x1", "x2"]
COLS_RUIDO = ["ruido1", "ruido2", "ruido3"]


def make_synthetic(
    n_samples=900,
    centros=((0.0, 0.0), (3.5, 0.0), (1.75, 3.0)),
    desviaciones=(0.6, 1.0, 1.4),
    frac_flip=0.10,
    frac_zona_frontera=0.25,
    n_ruido=3,
    n_outliers=15,
    centro_outliers=(9.0, -6.0),
    clase_outliers=0,
    seed=0,
):
    """Genera el dataset sintético con ruido estructurado.

    Parámetros
    ----------
    n_samples : total de muestras (incluye los outliers).
    centros : media de cada clase en los 2 rasgos informativos.
    desviaciones : desviación estándar POR CLASE (ruido heterocedástico).
    frac_flip : fracción del total de etiquetas que se voltean (~10%).
    frac_zona_frontera : el volteo solo se elige entre este porcentaje de
        puntos más cercanos a una frontera (los de menor "margen").
    n_ruido : número de rasgos irrelevantes N(0,1).
    n_outliers : tamaño del cúmulo de outliers.
    seed : semilla para reproducibilidad.

    Devuelve
    --------
    X (n, 2+n_ruido), y (etiquetas ya con ruido) y un DataFrame ``meta`` con
    la etiqueta verdadera y marcas de qué puntos fueron volteados / outliers.
    """
    rng = np.random.default_rng(seed)
    centros = np.asarray(centros, dtype=float)
    c = len(centros)
    n_reg = n_samples - n_outliers

    # --- 1) Clases gaussianas con varianza distinta por clase -------------
    # Reparto lo más equilibrado posible de las muestras regulares.
    conteos = np.full(c, n_reg // c)
    conteos[: n_reg % c] += 1
    y_verd = np.repeat(np.arange(c), conteos)
    X2 = np.empty((n_reg, 2))
    for k in range(c):
        idx = y_verd == k
        X2[idx] = rng.normal(centros[k], desviaciones[k], size=(idx.sum(), 2))

    # --- 2) Volteo de etiquetas SOLO cerca de las fronteras ---------------
    # Margen = (distancia al 2º centro más cercano) - (distancia al más cercano).
    # Margen pequeño => el punto está cerca de una frontera entre dos clases.
    d = np.linalg.norm(X2[:, None, :] - centros[None, :, :], axis=2)
    orden = np.argsort(d, axis=1)
    cerca1, cerca2 = orden[:, 0], orden[:, 1]
    margen = d[np.arange(n_reg), cerca2] - d[np.arange(n_reg), cerca1]

    n_flip = int(round(frac_flip * n_samples))
    n_candidatos = int(round(frac_zona_frontera * n_reg))
    candidatos = np.argsort(margen)[:n_candidatos]          # los más cercanos a la frontera
    elegidos = rng.choice(candidatos, size=n_flip, replace=False)

    y = y_verd.copy()
    # El punto pasa a la clase "vecina" (la 2ª más cercana): error plausible en la frontera.
    y[elegidos] = cerca2[elegidos]
    volteado = np.zeros(n_reg, dtype=bool)
    volteado[elegidos] = True

    # --- 3) Rasgos irrelevantes (ruido puro) ------------------------------
    Xr = rng.normal(0.0, 1.0, size=(n_reg, n_ruido))
    X = np.hstack([X2, Xr])

    # --- 4) Cúmulo de outliers lejano, compacto y con etiqueta fija -------
    Xo2 = rng.normal(centro_outliers, 0.3, size=(n_outliers, 2))
    Xo = np.hstack([Xo2, rng.normal(0.0, 1.0, size=(n_outliers, n_ruido))])
    X = np.vstack([X, Xo])
    y = np.concatenate([y, np.full(n_outliers, clase_outliers)])
    y_verd = np.concatenate([y_verd, np.full(n_outliers, clase_outliers)])
    volteado = np.concatenate([volteado, np.zeros(n_outliers, dtype=bool)])
    es_outlier = np.concatenate([np.zeros(n_reg, dtype=bool), np.ones(n_outliers, dtype=bool)])

    # Se barajan las filas para que el orden no revele la clase.
    perm = rng.permutation(len(y))
    meta = pd.DataFrame({"y_verdadera": y_verd, "volteada": volteado, "outlier": es_outlier}).iloc[perm]
    return X[perm], y[perm], meta.reset_index(drop=True)


def save_synthetic_csv(X, y, meta, ruta=None):
    """Guarda el dataset sintético en data/synthetic.csv."""
    ruta = Path(ruta) if ruta else RAIZ / "data" / "synthetic.csv"
    cols = COLS_INFORMATIVAS + [f"ruido{i+1}" for i in range(X.shape[1] - 2)]
    df = pd.DataFrame(X, columns=cols)
    df["y"] = y
    df = pd.concat([df, meta], axis=1)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)
    return ruta


def plot_synthetic(X, y, meta, ruta=None):
    """Figura del sintético: clases, etiquetas volteadas y outliers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ruta = Path(ruta) if ruta else RAIZ / "results" / "figures" / "synthetic_dataset.png"
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Izquierda: etiquetas observadas, marcando volteos y outliers
    for k in np.unique(y):
        m = (y == k) & ~meta["outlier"].values
        ax[0].scatter(X[m, 0], X[m, 1], s=14, c=colores[k], alpha=0.7, label=f"Clase {k}")
    v = meta["volteada"].values
    ax[0].scatter(X[v, 0], X[v, 1], s=60, facecolors="none", edgecolors="red", linewidths=1.2,
                  label="Etiqueta volteada")
    o = meta["outlier"].values
    ax[0].scatter(X[o, 0], X[o, 1], s=40, marker="x", c="black", label="Outliers")
    ax[0].set_title("Etiquetas observadas (con ruido)")

    # Derecha: etiquetas verdaderas, para ver que los volteos están en la frontera
    for k in np.unique(y):
        m = (meta["y_verdadera"].values == k) & ~o
        ax[1].scatter(X[m, 0], X[m, 1], s=14, c=colores[k], alpha=0.7, label=f"Clase {k}")
    ax[1].scatter(X[o, 0], X[o, 1], s=40, marker="x", c="black", label="Outliers")
    ax[1].set_title("Etiquetas verdaderas (antes del volteo)")

    for a in ax:
        a.set_xlabel("x1 (informativo)")
        a.set_ylabel("x2 (informativo)")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def load_wine_data():
    """Wine: 178 muestras, 13 rasgos numéricos, 3 clases."""
    from sklearn.datasets import load_wine
    d = load_wine()
    return d.data, d.target, list(d.feature_names)


def load_synthetic(seed=0):
    """Atajo: genera el sintético y devuelve solo (X, y)."""
    X, y, _ = make_synthetic(seed=seed)
    return X, y


if __name__ == "__main__":
    X, y, meta = make_synthetic(seed=0)
    print("CSV:", save_synthetic_csv(X, y, meta))
    print("Figura:", plot_synthetic(X, y, meta))
    print("Forma:", X.shape, "| clases:", np.bincount(y))
    print("Volteadas:", int(meta["volteada"].sum()), "| outliers:", int(meta["outlier"].sum()))
    Xw, yw, _ = load_wine_data()
    print("Wine:", Xw.shape, np.bincount(yw))
