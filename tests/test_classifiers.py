"""Pruebas: las implementaciones desde cero deben coincidir con sklearn.
sklearn se usa AQUÍ solo como referencia (oráculo)."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.classifiers import KNN, MinimumDistance, OneNN, pairwise_distances
from src.datasets import make_synthetic


def _conjuntos():
    """Dos escenarios: sintético (con ruido y outliers) y Wine escalado."""
    X, y, _ = make_synthetic(seed=3)
    yield "sintetico", train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    Xw, yw = load_wine(return_X_y=True)
    Xw = StandardScaler().fit_transform(Xw)
    yield "wine", train_test_split(Xw, yw, test_size=0.3, stratify=yw, random_state=0)


CONJUNTOS = list(_conjuntos())
IDS = [c[0] for c in CONJUNTOS]
DATOS = [c[1] for c in CONJUNTOS]


@pytest.mark.parametrize("datos", DATOS, ids=IDS)
def test_distancia_minima_igual_a_nearest_centroid(datos):
    Xtr, Xte, ytr, yte = datos
    propio = MinimumDistance().fit(Xtr, ytr)
    ref = NearestCentroid().fit(Xtr, ytr)
    np.testing.assert_array_equal(propio.predict(Xte), ref.predict(Xte))
    np.testing.assert_allclose(propio.centroids_, ref.centroids_)


@pytest.mark.parametrize("datos", DATOS, ids=IDS)
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_1nn_igual_a_sklearn(datos, metric):
    Xtr, Xte, ytr, yte = datos
    propio = OneNN(metric=metric).fit(Xtr, ytr)
    ref = KNeighborsClassifier(n_neighbors=1, metric=metric).fit(Xtr, ytr)
    np.testing.assert_array_equal(propio.predict(Xte), ref.predict(Xte))


@pytest.mark.parametrize("datos", DATOS, ids=IDS)
@pytest.mark.parametrize("k", [1, 3, 5, 11, 21])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_knn_igual_a_sklearn(datos, k, weights, metric):
    Xtr, Xte, ytr, yte = datos
    propio = KNN(k=k, weights=weights, metric=metric).fit(Xtr, ytr)
    ref = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric).fit(Xtr, ytr)
    np.testing.assert_array_equal(propio.predict(Xte), ref.predict(Xte))
    assert propio.score(Xte, yte) == pytest.approx(ref.score(Xte, yte))


def test_distancias_vectorizadas_correctas():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(7, 4)), rng.normal(size=(9, 4))
    esperado_e = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
    esperado_m = np.abs(A[:, None] - B[None]).sum(-1)
    np.testing.assert_allclose(pairwise_distances(A, B, "euclidean"), esperado_e, atol=1e-9)
    np.testing.assert_allclose(pairwise_distances(A, B, "manhattan"), esperado_m)


def test_desempate_determinista():
    # Dos vecinos de clases distintas a igual distancia: gana la clase menor, siempre.
    X = np.array([[0.0], [2.0]])
    y = np.array([1, 0])
    modelo = KNN(k=2).fit(X, y)
    assert modelo.predict([[1.0]])[0] == 0
    assert modelo.predict([[1.0]])[0] == modelo.predict([[1.0]])[0]


def test_k_mayor_que_n_no_falla():
    X, y = np.array([[0.0], [1.0], [5.0]]), np.array([0, 0, 1])
    assert KNN(k=50).fit(X, y).predict([[0.2]])[0] == 0


def test_interfaz_sklearn_pipeline_y_gridsearch():
    X, y = load_wine(return_X_y=True)
    for modelo in (MinimumDistance(), OneNN(), KNN()):
        assert isinstance(clone(modelo), type(modelo))
    pipe = Pipeline([("sc", StandardScaler()), ("clf", KNN())])
    gs = GridSearchCV(pipe, {"clf__k": [1, 5], "clf__weights": ["uniform", "distance"]}, cv=3)
    gs.fit(X, y)
    assert gs.best_score_ > 0.9
    assert 0.0 <= OneNN().set_params(metric="manhattan").fit(X, y).score(X, y) <= 1.0
