"""Fábrica de los 6 clasificadores, todos con parámetros POR DEFECTO.

Cada modelo se devuelve dentro de un Pipeline [StandardScaler -> clasificador].
Al usar Pipeline, el escalador se ajusta SOLO con el train de cada partición
(así no se filtra información del test al entrenamiento: evita "data leakage").
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.classifiers import KNN, MinimumDistance, OneNN

NOMBRES = ["Distancia Mínima", "1-NN", "K-NN", "Naive Bayes", "SVM", "MLP"]


def crear_clasificadores(seed=0):
    """Devuelve {nombre: clasificador} con los hiperparámetros por defecto."""
    return {
        "Distancia Mínima": MinimumDistance(),   # desde cero
        "1-NN": OneNN(),                         # desde cero
        "K-NN": KNN(),                           # desde cero (k=5, uniforme, euclídea)
        "Naive Bayes": GaussianNB(),             # sklearn
        "SVM": SVC(),                            # sklearn (kernel rbf, C=1)
        "MLP": MLPClassifier(random_state=seed), # sklearn (100 neuronas ocultas, relu, max_iter=200)
    }


def crear_modelo(nombre, seed=0, escalar=True):
    """Pipeline listo para validar. Con escalar=False se omite el StandardScaler
    (se usa en el estudio del efecto del escalado)."""
    clf = crear_clasificadores(seed)[nombre]
    pasos = ([("escalador", StandardScaler())] if escalar else []) + [("clf", clf)]
    return Pipeline(pasos)
