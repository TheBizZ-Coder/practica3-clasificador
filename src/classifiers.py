"""Clasificadores implementados DESDE CERO con NumPy.

- MinimumDistance : distancia mínima al centroide de cada clase.
- OneNN           : 1 vecino más cercano.
- KNN             : k vecinos más cercanos (voto uniforme o ponderado por distancia).

Heredan de BaseEstimator (get_params/set_params/clone) y ClassifierMixin (score),
así que funcionan dentro de Pipeline y GridSearchCV.
Aquí NO se usa sklearn.neighbors ni nada parecido: solo NumPy para el cálculo.
(sklearn se importa únicamente por las clases base y las validaciones de entrada.)
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

# Tamaño máximo (en nº de elementos) del tensor intermedio de Manhattan.
# Se procesa por bloques de filas para no agotar la memoria.
_MAX_ELEMS = 20_000_000


def pairwise_distances(A, B, metric="euclidean"):
    """Matriz de distancias (m, n) entre filas de A (m,d) y filas de B (n,d), VECTORIZADA.

    - euclidean: usa la identidad ||a-b||² = ||a||² + ||b||² - 2·a·b,
      que se reduce a un producto matricial (muy rápido con BLAS).
    - manhattan: suma de |a-b| usando broadcasting, por bloques de filas.
    """
    if metric == "euclidean":
        a2 = np.sum(A * A, axis=1)[:, None]          # (m,1)
        b2 = np.sum(B * B, axis=1)[None, :]          # (1,n)
        d2 = a2 + b2 - 2.0 * (A @ B.T)               # (m,n)
        np.maximum(d2, 0.0, out=d2)                  # errores numéricos pueden dar -1e-16
        return np.sqrt(d2)
    if metric == "manhattan":
        m, n, d = A.shape[0], B.shape[0], A.shape[1]
        D = np.empty((m, n))
        bloque = max(1, _MAX_ELEMS // max(1, n * d))
        for i in range(0, m, bloque):
            D[i:i + bloque] = np.abs(A[i:i + bloque, None, :] - B[None, :, :]).sum(axis=2)
        return D
    raise ValueError(f"metric debe ser 'euclidean' o 'manhattan', no {metric!r}")


class MinimumDistance(ClassifierMixin, BaseEstimator):
    """Clasificador de distancia mínima: asigna la clase del centroide más cercano."""

    def __init__(self, metric="euclidean"):
        self.metric = metric

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        # classes_ ordenadas; y_idx = posición de cada etiqueta dentro de classes_
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        # Centroide = media de las muestras de cada clase (una fila por clase)
        self.centroids_ = np.vstack([X[y_idx == k].mean(axis=0) for k in range(len(self.classes_))])
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        D = pairwise_distances(X, self.centroids_, self.metric)   # (m, c)
        # argmin devuelve el primer mínimo => desempate determinista (clase menor)
        return self.classes_[np.argmin(D, axis=1)]


class KNN(ClassifierMixin, BaseEstimator):
    """K vecinos más cercanos.

    Parámetros
    ----------
    k : número de vecinos (si k > n_train se usa n_train).
    weights : 'uniform' (cada vecino 1 voto) | 'distance' (voto = 1/distancia).
    metric : 'euclidean' | 'manhattan'.

    Desempates (deterministas):
      * entre vecinos a igual distancia gana el de menor índice de entrenamiento
        (argsort estable);
      * entre clases con igual voto gana la de menor etiqueta (argmax).
    """

    def __init__(self, k=5, weights="uniform", metric="euclidean"):
        self.k = k
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        # KNN es "perezoso": fit solo memoriza los datos (O(n·d) memoria, sin cómputo).
        X, y = check_X_y(X, y)
        if self.weights not in ("uniform", "distance"):
            raise ValueError("weights debe ser 'uniform' o 'distance'")
        self.classes_, self._y_idx = np.unique(y, return_inverse=True)
        self._X = X
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        k = min(self.k, self._X.shape[0])
        D = pairwise_distances(X, self._X, self.metric)             # (m, n)
        # Índices de los k vecinos más cercanos. 'stable' fija el desempate por índice.
        orden = np.argsort(D, axis=1, kind="stable")[:, :k]         # (m, k)
        dk = np.take_along_axis(D, orden, axis=1)                   # distancias de los vecinos
        etiquetas = self._y_idx[orden]                              # clase de cada vecino

        if self.weights == "uniform":
            w = np.ones_like(dk)
        else:
            # Peso 1/d. Si algún vecino está a distancia 0 (punto idéntico),
            # solo cuentan esos (convención habitual): su peso es 1 y el resto 0.
            cero = dk == 0
            hay_cero = cero.any(axis=1, keepdims=True)
            with np.errstate(divide="ignore"):
                w = np.where(hay_cero, cero.astype(float), 1.0 / dk)

        # Votos por clase: suma de pesos de los vecinos de cada clase -> (m, c)
        votos = np.zeros((X.shape[0], len(self.classes_)))
        for c in range(len(self.classes_)):
            votos[:, c] = (w * (etiquetas == c)).sum(axis=1)
        return self.classes_[np.argmax(votos, axis=1)]


class OneNN(KNN):
    """1-NN: caso particular de KNN con k=1 (se expone sin parámetro k)."""

    def __init__(self, metric="euclidean"):
        super().__init__(k=1, weights="uniform", metric=metric)

    def get_params(self, deep=True):
        # Solo 'metric' es configurable; así clone() y GridSearchCV funcionan bien.
        return {"metric": self.metric}

    def set_params(self, **params):
        for nombre, valor in params.items():
            if nombre != "metric":
                raise ValueError(f"OneNN solo admite el parámetro 'metric', no {nombre!r}")
            self.metric = valor
        return self
