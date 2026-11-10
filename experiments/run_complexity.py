"""Fase 5: tabla teórica de complejidad + experimento empírico (tiempos vs n y vs d)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.complexity import (TABLAS, estimar_pendientes, experimento_empirico, graficar,
                            tabla_teorica)


def ejecutar(seed=0):
    TABLAS.mkdir(parents=True, exist_ok=True)
    teo = tabla_teorica()
    teo.to_csv(TABLAS / "complejidad_teorica.csv", index=False)
    (TABLAS / "complejidad_teorica.md").write_text(teo.to_markdown(index=False), encoding="utf-8")

    df = experimento_empirico(seed=seed)
    df.to_csv(TABLAS / "complejidad_empirica.csv", index=False)
    pend = estimar_pendientes(df)
    pend.to_csv(TABLAS / "complejidad_pendientes.csv", index=False)
    (TABLAS / "complejidad_pendientes.md").write_text(pend.to_markdown(index=False), encoding="utf-8")
    graficar(df, pend)
    print(pend.to_string(index=False))


if __name__ == "__main__":
    ejecutar(seed=0)
