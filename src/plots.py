"""Gráficas: matrices de confusión y fronteras de decisión."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.models import crear_modelo

RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = RAIZ / "results" / "figures"
COLORES = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def plot_matrices_confusion(cms, clases, titulo, ruta):
    """Una figura con la matriz de confusión agregada de cada modelo (2x3)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (nombre, cm) in zip(axes.ravel(), cms.items()):
        ax.imshow(cm, cmap="Blues")
        ax.set_title(nombre)
        ax.set_xlabel("Predicha")
        ax.set_ylabel("Real")
        ax.set_xticks(range(len(clases)), clases)
        ax.set_yticks(range(len(clases)), clases)
        for i in range(len(clases)):
            for j in range(len(clases)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle(titulo)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def plot_fronteras(X2, y, nombres, ruta, seed=0, titulo="Fronteras de decisión (parámetros por defecto)"):
    """Fronteras de decisión de los 6 modelos usando SOLO los 2 rasgos informativos.

    Cada modelo (con su StandardScaler) se entrena con (x1, x2) y se evalúa en una
    malla de puntos; cada región se colorea con la clase predicha.
    """
    # Se enfoca el gráfico en la nube principal (los outliers quedan fuera del recuadro).
    xmin, xmax, ymin, ymax = -3.5, 7.0, -3.5, 7.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 250), np.linspace(ymin, ymax, 250))
    malla = np.c_[xx.ravel(), yy.ravel()]
    claro = ListedColormap(["#aec7e8", "#ffbb78", "#98df8a"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, nombre in zip(axes.ravel(), nombres):
        modelo = crear_modelo(nombre, seed).fit(X2, y)
        zz = modelo.predict(malla).reshape(xx.shape)
        ax.contourf(xx, yy, zz, cmap=claro, alpha=0.8, levels=[-0.5, 0.5, 1.5, 2.5])
        for k in np.unique(y):
            ax.scatter(X2[y == k, 0], X2[y == k, 1], s=8, c=COLORES[k], edgecolors="none")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{nombre} (acc. entrenamiento = {modelo.score(X2, y):.3f})", fontsize=10)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    fig.suptitle(titulo)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
