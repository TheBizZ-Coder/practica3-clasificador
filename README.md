# Práctica 3 – Comparación de clasificadores

Comparación de 6 clasificadores: Distancia Mínima, 1-NN, K-NN (implementados desde cero con NumPy),
Naïve Bayes gaussiano, SVM y MLP (scikit-learn), sobre un dataset sintético con ruido estructurado y Wine.

## Estructura
- `src/`: datasets, clasificadores, validación, gráficas, complejidad y ajuste de hiperparámetros.
- `experiments/`: scripts que ejecutan cada fase (`run_all.py` reproduce todo).
- `tests/`: pruebas pytest (comparan contra sklearn como referencia).
- `data/`, `results/tables`, `results/figures`: datos y resultados generados.
- `report/reporte.md`: reporte final.
- `ia_conversacion/`: registro de la conversación con la IA usada.

## Uso
```
pip install -r requirements.txt
pytest
python experiments/run_all.py
```
