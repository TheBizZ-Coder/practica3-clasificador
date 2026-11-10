"""Fase 4: 6 modelos x 2 datasets x 2 validaciones, con parámetros POR DEFECTO.

Genera en results/tables: defaults_por_particion.csv, defaults_resumen.csv/.md,
defaults_sin_escalado.csv/.md (estudio del escalado) y matrices de confusión;
en results/figures: matrices de confusión y fronteras de decisión.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.datasets import load_wine_data, make_synthetic
from src.models import NOMBRES, crear_modelo
from src.plots import FIGURAS, plot_fronteras, plot_matrices_confusion
from src.validation import crear_particiones, evaluar, resumir

TABLAS = Path(__file__).resolve().parent.parent / "results" / "tables"


def cargar_datasets(seed):
    Xs, ys, _ = make_synthetic(seed=seed)
    Xw, yw, _ = load_wine_data()
    return {"sintetico": (Xs, ys), "wine": (Xw, yw)}


def guardar_tabla(df, nombre):
    """Guarda en CSV y en Markdown."""
    TABLAS.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS / f"{nombre}.csv", index=False)
    (TABLAS / f"{nombre}.md").write_text(df.to_markdown(index=False), encoding="utf-8")


def formato_resumen(df):
    """Tabla legible 'media ± std' a partir del resumen numérico."""
    out = df[["dataset", "validacion", "modelo"]].copy()
    for c, nombre, dec in [("accuracy", "Accuracy", 4), ("f1_macro", "F1 macro", 4),
                           ("t_fit_s", "t fit (s)", 5), ("t_predict_s", "t predict (s)", 5)]:
        out[nombre] = [f"{m:.{dec}f} ± {s:.{dec}f}" for m, s in zip(df[f"{c}_media"], df[f"{c}_std"])]
    return out


def ejecutar(seed=0):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)  # MLP con max_iter=200 por defecto
    datasets = cargar_datasets(seed)
    por_particion, resumenes, sin_escala = [], [], []

    for ds, (X, y) in datasets.items():
        clases = np.unique(y)
        for val in ["holdout", "kfold"]:
            # Mismas particiones para todos los modelos
            particiones = crear_particiones(X, y, val, seed)
            cms = {}
            for nombre in NOMBRES:
                df, cm = evaluar(crear_modelo(nombre, seed), X, y, particiones, clases)
                cms[nombre] = cm
                df.insert(0, "modelo", nombre)
                df.insert(0, "validacion", val)
                df.insert(0, "dataset", ds)
                por_particion.append(df)
                resumenes.append({"dataset": ds, "validacion": val, "modelo": nombre, **resumir(df)})
                pd.DataFrame(cm, index=[f"real_{c}" for c in clases],
                             columns=[f"pred_{c}" for c in clases]).to_csv(
                    TABLAS / f"confusion_{ds}_{val}_{nombre.replace(' ', '_')}.csv")
                # Estudio del escalado: mismo modelo SIN StandardScaler
                df2, _ = evaluar(crear_modelo(nombre, seed, escalar=False), X, y, particiones, clases)
                sin_escala.append({"dataset": ds, "validacion": val, "modelo": nombre, **resumir(df2)})
            plot_matrices_confusion(
                cms, clases, f"Matrices de confusión agregadas – {ds} – {val}",
                FIGURAS / f"confusion_{ds}_{val}.png")
            print(f"  listo: {ds} / {val}")

    pd.concat(por_particion).to_csv(TABLAS / "defaults_por_particion.csv", index=False)
    res = pd.DataFrame(resumenes)
    res.to_csv(TABLAS / "defaults_resumen_numerico.csv", index=False)
    guardar_tabla(formato_resumen(res), "defaults_resumen")

    sin = pd.DataFrame(sin_escala)
    sin.to_csv(TABLAS / "defaults_sin_escalado_numerico.csv", index=False)
    # Comparación con/sin escalado (solo accuracy)
    comp = res[["dataset", "validacion", "modelo", "accuracy_media"]].rename(
        columns={"accuracy_media": "acc_con_escalado"})
    comp["acc_sin_escalado"] = sin["accuracy_media"].values
    comp["diferencia"] = comp["acc_con_escalado"] - comp["acc_sin_escalado"]
    guardar_tabla(comp.round(4), "defaults_efecto_escalado")

    # Fronteras de decisión en el sintético, solo con los 2 rasgos informativos
    Xs, ys = datasets["sintetico"]
    plot_fronteras(Xs[:, :2], ys, NOMBRES, FIGURAS / "fronteras_sintetico.png", seed)
    print(formato_resumen(res).to_string(index=False))
    return res


if __name__ == "__main__":
    ejecutar(seed=0)
