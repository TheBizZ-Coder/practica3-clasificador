# Comparación de clasificadores: Distancia Mínima, 1-NN, K-NN, Naive Bayes, SVM y MLP

*Borrador del reporte – Práctica 3. Todas las cifras provienen de los CSV de `results/tables/` (semilla global 0).*

## 1. Introducción

El objetivo es comparar seis clasificadores supervisados sobre dos conjuntos de datos: uno **sintético con ruido estructurado**, diseñado para estresar a los modelos, y el **benchmark Wine** (178 muestras, 13 rasgos, 3 clases). Distancia Mínima, 1-NN y K-NN se implementaron **desde cero con NumPy** (`src/classifiers.py`) y se verificaron contra `NearestCentroid` y `KNeighborsClassifier` de scikit-learn (50 pruebas pytest, todas pasan). Naive Bayes gaussiano, SVM y MLP se usaron de scikit-learn.

Se responden tres preguntas: (1) ¿cómo se comparan los modelos con parámetros por defecto?, (2) ¿cuánto cuestan en tiempo y cómo escala ese costo?, y (3) ¿mejora el ajuste de hiperparámetros los resultados, y de forma significativa?

## 2. Marco teórico

- **Distancia Mínima.** Cada clase se resume por su centroide (media); un punto se asigna a la clase del centroide más cercano. Produce fronteras lineales.
- **1-NN y K-NN** (Cover y Hart, 1967). Clasificadores perezosos: memorizan el entrenamiento y votan entre los *k* vecinos más cercanos (voto uniforme o ponderado por 1/distancia; distancia euclídea o Manhattan). *k* controla el sesgo/varianza: *k* pequeño da fronteras muy irregulares.
- **Naive Bayes gaussiano.** Aplica Bayes suponiendo rasgos independientes dados la clase y gaussianos; estima media y varianza por clase y rasgo.
- **SVM** (Cortes y Vapnik, 1995). Busca el hiperplano de máximo margen, posiblemente en un espacio transformado por un kernel (lineal, RBF, polinomial). *C* regula la tolerancia a errores y *gamma* el alcance del kernel RBF.
- **MLP.** Red neuronal de capas densas entrenada con retropropagación; sus hiperparámetros incluyen arquitectura, activación, regularización L2 (`alpha`) y tasa de aprendizaje.

### Complejidad teórica
Con *n* muestras de entrenamiento, *m* de prueba, *d* rasgos, *c* clases, *k* vecinos, *n_sv* vectores de soporte, *W* pesos y *T* épocas:

| Modelo                   | Entrenamiento          | Predicción           | Memoria   | Comentario                                                                                    |
|:-------------------------|:-----------------------|:---------------------|:----------|:----------------------------------------------------------------------------------------------|
| Distancia Mínima         | O(n·d)                 | O(m·c·d)             | O(c·d)    | Entrenar = calcular c medias; predecir = distancia a c centroides.                            |
| 1-NN                     | O(n·d) (solo almacena) | O(m·n·d)             | O(n·d)    | Compara cada punto de prueba contra los n de entrenamiento.                                   |
| K-NN                     | O(n·d) (solo almacena) | O(m·n·d + m·n·log k) | O(n·d)    | Igual que 1-NN más la selección de los k menores (nuestra versión ordena todo: O(m·n·log n)). |
| Naive Bayes              | O(n·d)                 | O(m·c·d)             | O(c·d)    | Media y varianza por clase y rasgo; predecir evalúa c·d gaussianas.                           |
| SVM (kernel RBF, libsvm) | entre O(n²·d) y O(n³)  | O(m·n_sv·d)          | O(n_sv·d) | Resuelve un problema cuadrático; la predicción depende de los vectores de soporte.            |
| MLP                      | O(T·n·W)               | O(m·W)               | O(W)      | Cada época recorre los n datos con coste proporcional a W (propagación + retropropagación).   |

## 3. Metodología

**Dataset sintético** (`src/datasets.py`, 900 muestras): tres gaussianas en dos rasgos informativos con desviaciones distintas por clase (0.6, 1.0 y 1.4: ruido heterocedástico); 90 etiquetas (10 %) volteadas **solo entre los puntos más cercanos a las fronteras** (el 25 % de menor margen), que pasan a la clase vecina; tres rasgos irrelevantes N(0,1); y un cúmulo de 15 outliers lejanos. Los 15 outliers forman parte de las 900 muestras. Datos en `data/synthetic.csv`; figura en `results/figures/synthetic_dataset.png`.

**Validación** (`src/validation.py`): (a) Hold-Out 70/30 estratificado repetido con 10 semillas; (b) K-Fold estratificado con k = 10. Las particiones se generan una vez y se reutilizan en todos los modelos. Cada modelo es un `Pipeline` con `StandardScaler` + clasificador, de modo que el escalador se ajusta solo con el train de cada partición. Métricas: accuracy, F1 macro, matriz de confusión agregada y tiempos de fit y predict. Se reporta media ± desviación estándar sobre las particiones.

**Parámetros por defecto:** K-NN con k = 5, pesos uniformes y distancia euclídea; SVM con kernel RBF y C = 1; MLP con una capa oculta de 100 neuronas, ReLU y `max_iter = 200` (esto produce avisos de convergencia en algunos casos, que se silenciaron).

**Ajuste de hiperparámetros** (`src/tuning.py`): validación cruzada **anidada**, con CV interno de 5 folds para elegir parámetros y CV externo de 10 folds (los mismos folds de la validación por defecto) para estimar el desempeño. K-NN usa `GridSearchCV` (k ∈ {1, 3, …, 21} × pesos × métrica); SVM (kernel ∈ {linear, rbf, poly}, C y gamma log-uniformes) y MLP (arquitectura, activación, `alpha` y `learning_rate_init` log-uniformes, `early_stopping`) usan `RandomizedSearchCV` con 40 y 20 iteraciones, respectivamente. La comparación entre default y ajustado usa la prueba de Wilcoxon pareada sobre los 10 folds externos.

## 4. Resultados con parámetros por defecto

| dataset   | validacion   | modelo           | Accuracy        | F1 macro        | t fit (s)         | t predict (s)     |
|:----------|:-------------|:-----------------|:----------------|:----------------|:------------------|:------------------|
| sintetico | holdout      | Distancia Mínima | 0.8374 ± 0.0227 | 0.8347 ± 0.0237 | 0.00100 ± 0.00016 | 0.00031 ± 0.00004 |
| sintetico | holdout      | 1-NN             | 0.8067 ± 0.0153 | 0.8034 ± 0.0160 | 0.00092 ± 0.00006 | 0.01099 ± 0.00029 |
| sintetico | holdout      | K-NN             | 0.8544 ± 0.0257 | 0.8516 ± 0.0262 | 0.00092 ± 0.00005 | 0.01149 ± 0.00111 |
| sintetico | holdout      | Naive Bayes      | 0.8456 ± 0.0222 | 0.8418 ± 0.0237 | 0.00150 ± 0.00009 | 0.00041 ± 0.00008 |
| sintetico | holdout      | SVM              | 0.8615 ± 0.0264 | 0.8589 ± 0.0271 | 0.00711 ± 0.00169 | 0.00475 ± 0.00130 |
| sintetico | holdout      | MLP              | 0.8589 ± 0.0226 | 0.8564 ± 0.0232 | 0.34267 ± 0.01417 | 0.00071 ± 0.00019 |
| sintetico | kfold        | Distancia Mínima | 0.8300 ± 0.0413 | 0.8270 ± 0.0429 | 0.00104 ± 0.00014 | 0.00030 ± 0.00001 |
| sintetico | kfold        | 1-NN             | 0.8122 ± 0.0394 | 0.8085 ± 0.0399 | 0.00090 ± 0.00005 | 0.00421 ± 0.00010 |
| sintetico | kfold        | K-NN             | 0.8533 ± 0.0362 | 0.8504 ± 0.0380 | 0.00095 ± 0.00019 | 0.00427 ± 0.00033 |
| sintetico | kfold        | Naive Bayes      | 0.8433 ± 0.0407 | 0.8395 ± 0.0425 | 0.00164 ± 0.00027 | 0.00038 ± 0.00005 |
| sintetico | kfold        | SVM              | 0.8600 ± 0.0410 | 0.8574 ± 0.0437 | 0.00848 ± 0.00138 | 0.00193 ± 0.00009 |
| sintetico | kfold        | MLP              | 0.8567 ± 0.0361 | 0.8543 ± 0.0380 | 0.41852 ± 0.03043 | 0.00059 ± 0.00008 |
| wine      | holdout      | Distancia Mínima | 0.9630 ± 0.0175 | 0.9645 ± 0.0166 | 0.00096 ± 0.00021 | 0.00030 ± 0.00003 |
| wine      | holdout      | 1-NN             | 0.9574 ± 0.0248 | 0.9585 ± 0.0239 | 0.00096 ± 0.00013 | 0.00069 ± 0.00011 |
| wine      | holdout      | K-NN             | 0.9500 ± 0.0196 | 0.9515 ± 0.0181 | 0.00095 ± 0.00008 | 0.00067 ± 0.00004 |
| wine      | holdout      | Naive Bayes      | 0.9741 ± 0.0096 | 0.9750 ± 0.0093 | 0.00153 ± 0.00009 | 0.00041 ± 0.00007 |
| wine      | holdout      | SVM              | 0.9852 ± 0.0191 | 0.9853 ± 0.0189 | 0.00175 ± 0.00010 | 0.00053 ± 0.00002 |
| wine      | holdout      | MLP              | 0.9722 ± 0.0235 | 0.9726 ± 0.0232 | 0.08139 ± 0.00094 | 0.00054 ± 0.00001 |
| wine      | kfold        | Distancia Mínima | 0.9719 ± 0.0396 | 0.9719 ± 0.0397 | 0.00090 ± 0.00010 | 0.00032 ± 0.00009 |
| wine      | kfold        | 1-NN             | 0.9552 ± 0.0439 | 0.9569 ± 0.0425 | 0.00129 ± 0.00033 | 0.00065 ± 0.00014 |
| wine      | kfold        | K-NN             | 0.9608 ± 0.0271 | 0.9617 ± 0.0267 | 0.00088 ± 0.00011 | 0.00047 ± 0.00003 |
| wine      | kfold        | Naive Bayes      | 0.9719 ± 0.0296 | 0.9728 ± 0.0288 | 0.00144 ± 0.00012 | 0.00035 ± 0.00001 |
| wine      | kfold        | SVM              | 0.9830 ± 0.0274 | 0.9828 ± 0.0277 | 0.00199 ± 0.00048 | 0.00046 ± 0.00010 |
| wine      | kfold        | MLP              | 0.9778 ± 0.0287 | 0.9778 ± 0.0288 | 0.10078 ± 0.01118 | 0.00059 ± 0.00015 |

**Dataset sintético.** Con K-Fold, el orden por accuracy es SVM (0.8600), MLP (0.8567), K-NN (0.8533), Naive Bayes (0.8433), Distancia Mínima (0.8300) y 1-NN (0.8122); con Hold-Out el orden es el mismo (SVM 0.8615, MLP 0.8589, K-NN 0.8544, Naive Bayes 0.8456, Distancia Mínima 0.8374, 1-NN 0.8067). Las desviaciones estándar (0.015 a 0.041) son del mismo orden que varias de las diferencias entre modelos, así que las diferencias entre SVM, MLP y K-NN no son concluyentes. Como el 10 % de las etiquetas está volteado, ningún modelo supera el 0.87 de accuracy.

**Wine.** Con K-Fold, SVM (0.9830) lidera, seguido de MLP (0.9778), Naive Bayes y Distancia Mínima (ambos 0.9719), K-NN (0.9608) y 1-NN (0.9552). Con Hold-Out el orden es SVM (0.9852), Naive Bayes (0.9741), MLP (0.9722), Distancia Mínima (0.9630), 1-NN (0.9574) y K-NN (0.9500). Las desviaciones (0.010 a 0.044) tapan casi todas las diferencias.

**Tiempos.** MLP es el más lento en entrenamiento (≈ 0.34 a 0.42 s en el sintético, ≈ 0.08 a 0.10 s en Wine, frente a ≈ 0.001 a 0.009 s del resto); 1-NN y K-NN son los más lentos al predecir en el sintético (≈ 0.011 s por Hold-Out de 270 muestras y ≈ 0.004 s por fold de 90), frente a ≈ 0.0003 a 0.0005 s de Distancia Mínima y Naive Bayes.

**Matrices de confusión.** Las figuras están en `results/figures/confusion_<dataset>_<validación>.png`. En el sintético con K-Fold, 1-NN comete 169 errores de 900 y K-NN 132; los errores se concentran entre las clases 1 y 2 y entre las 0 y 2 (los dos pares de clases vecinas con más solapamiento). En Wine, los errores de 1-NN y K-NN se limitan casi por completo a la clase 1.

**Fronteras de decisión** (`results/figures/fronteras_sintetico.png`, solo con x1 y x2). Distancia Mínima produce fronteras rectas; Naive Bayes, curvas suaves; SVM y MLP, fronteras suaves y flexibles; K-NN, fronteras irregulares; 1-NN reproduce cada etiqueta de entrenamiento (accuracy de entrenamiento = 1.000), por lo que genera islas alrededor de las etiquetas volteadas.

### Efecto del escalado (mismo experimento sin `StandardScaler`)

| dataset   | validacion   | modelo           |   acc_con_escalado |   acc_sin_escalado |   diferencia |
|:----------|:-------------|:-----------------|-------------------:|-------------------:|-------------:|
| sintetico | holdout      | Distancia Mínima |             0.8374 |             0.8337 |       0.0037 |
| sintetico | holdout      | 1-NN             |             0.8067 |             0.8352 |      -0.0285 |
| sintetico | holdout      | K-NN             |             0.8544 |             0.8581 |      -0.0037 |
| sintetico | holdout      | Naive Bayes      |             0.8456 |             0.8456 |       0      |
| sintetico | holdout      | SVM              |             0.8615 |             0.8659 |      -0.0044 |
| sintetico | holdout      | MLP              |             0.8589 |             0.8519 |       0.007  |
| sintetico | kfold        | Distancia Mínima |             0.83   |             0.8278 |       0.0022 |
| sintetico | kfold        | 1-NN             |             0.8122 |             0.8411 |      -0.0289 |
| sintetico | kfold        | K-NN             |             0.8533 |             0.8611 |      -0.0078 |
| sintetico | kfold        | Naive Bayes      |             0.8433 |             0.8433 |       0      |
| sintetico | kfold        | SVM              |             0.86   |             0.8633 |      -0.0033 |
| sintetico | kfold        | MLP              |             0.8567 |             0.8522 |       0.0044 |
| wine      | holdout      | Distancia Mínima |             0.963  |             0.7    |       0.263  |
| wine      | holdout      | 1-NN             |             0.9574 |             0.7463 |       0.2111 |
| wine      | holdout      | K-NN             |             0.95   |             0.687  |       0.263  |
| wine      | holdout      | Naive Bayes      |             0.9741 |             0.9741 |       0      |
| wine      | holdout      | SVM              |             0.9852 |             0.6389 |       0.3463 |
| wine      | holdout      | MLP              |             0.9722 |             0.9    |       0.0722 |
| wine      | kfold        | Distancia Mínima |             0.9719 |             0.7196 |       0.2523 |
| wine      | kfold        | 1-NN             |             0.9552 |             0.7637 |       0.1915 |
| wine      | kfold        | K-NN             |             0.9608 |             0.6748 |       0.2859 |
| wine      | kfold        | Naive Bayes      |             0.9719 |             0.9719 |       0      |
| wine      | kfold        | SVM              |             0.983  |             0.6627 |       0.3203 |
| wine      | kfold        | MLP              |             0.9778 |             0.9147 |       0.0631 |

## 5. Análisis de complejidad

Tiempos medidos con *m* = 1000 muestras de prueba; vs. *n*: *d* = 5 y *n* de 500 a 20 000; vs. *d*: *n* = 2000 y *d* de 2 a 200 (rasgos extra de ruido). Las gráficas log-log están en `results/figures/complejidad_vs_n.png` y `complejidad_vs_d.png`. La pendiente se estimó por regresión lineal de log10(t) sobre log10(x), usando todos los puntos y solo la mitad superior (donde dominan los costos reales sobre los fijos).

| variable   | modelo           | etapa   |   pendiente_todos |   pendiente_mitad_superior | pendiente_teorica   |
|:-----------|:-----------------|:--------|------------------:|---------------------------:|:--------------------|
| n          | Distancia Mínima | fit     |             0.372 |                      0.79  | 1                   |
| n          | Distancia Mínima | predict |            -0.029 |                      0.202 | 0                   |
| n          | 1-NN             | fit     |             0.016 |                     -0.067 | 1                   |
| n          | 1-NN             | predict |             1.148 |                      1.067 | 1                   |
| n          | K-NN             | fit     |             0.179 |                      0.273 | 1                   |
| n          | K-NN             | predict |             1.123 |                      1.123 | 1                   |
| n          | Naive Bayes      | fit     |             0.457 |                      0.826 | 1                   |
| n          | Naive Bayes      | predict |             0.057 |                      0.047 | 0                   |
| n          | SVM              | fit     |             1.759 |                      1.9   | 2 a 3               |
| n          | SVM              | predict |             0.897 |                      0.942 | 1                   |
| n          | MLP              | fit     |             0.967 |                      0.987 | 1                   |
| n          | MLP              | predict |             0.084 |                      0.12  | 0                   |
| d          | Distancia Mínima | fit     |             0.231 |                      0.599 | 1                   |
| d          | Distancia Mínima | predict |             0.295 |                      0.704 | 1                   |
| d          | 1-NN             | fit     |             0.157 |                      0.285 | 1                   |
| d          | 1-NN             | predict |             0.002 |                      0.005 | 1                   |
| d          | K-NN             | fit     |             0.139 |                      0.271 | 1                   |
| d          | K-NN             | predict |             0.012 |                      0.002 | 1                   |
| d          | Naive Bayes      | fit     |             0.311 |                      0.716 | 1                   |
| d          | Naive Bayes      | predict |             0.535 |                      1.116 | 1                   |
| d          | SVM              | fit     |             0.59  |                      0.636 | 1                   |
| d          | SVM              | predict |             0.463 |                      0.399 | 1                   |
| d          | MLP              | fit     |             0.228 |                      0.362 | 1                   |
| d          | MLP              | predict |             0.125 |                      0.102 | 1                   |

Observaciones (las pendientes de tiempos de milisegundos son ruidosas):

- **Variando n.** La predicción de 1-NN y K-NN crece de forma casi lineal (pendientes 1.15 y 1.12), tal como predice O(m·n·d); con n = 20 000 tardan ≈ 2 s en predecir 1000 muestras. El fit de ambos es ≈ constante (pendientes ≈ 0.0 a 0.3): solo memorizan. SVM entrena con pendiente 1.76 (todos) y 1.90 (mitad superior), algo por debajo de los exponentes teóricos 2 a 3, y su predicción crece con pendiente 0.90 a 0.94, coherente con que *n_sv* crece con *n*. MLP entrena con pendiente ≈ 0.97 a 0.99 (lineal en *n* a épocas fijas), y su predicción apenas depende de *n* (pendiente 0.08 a 0.12). Distancia Mínima y Naive Bayes, con costos de microsegundos, tienen pendientes de fit menores a 1 en el ajuste global (0.37 y 0.46) porque dominan los costos fijos; en la mitad superior se acercan a 1 (0.79 y 0.83).
- **Variando d.** Las pendientes observadas quedan por debajo de la teoría para casi todos los modelos con *d* pequeño, porque los costos fijos dominan y BLAS vectoriza el producto matricial. En 1-NN y K-NN la predicción casi no depende de *d* (pendientes ≈ 0.00 a 0.01): con *n* = 2000 y *m* = 1000 domina la **ordenación** de la matriz *m × n* de distancias (O(m·n·log n)), que no depende de *d*. Naive Bayes sí muestra la dependencia lineal en predicción (0.54 global, 1.12 en la mitad superior). SVM sube con pendiente 0.59 a 0.64 en fit, y MLP con 0.23 a 0.36, porque solo la primera capa crece con *d*.
- Con *n* = 20 000, el entrenamiento de MLP (≈ 9.6 s) y de SVM (≈ 2.5 s) supera por mucho al de los demás modelos (≤ 0.005 s), mientras que 1-NN y K-NN pasan el costo a la predicción (≈ 2 s).

## 6. Ajuste de hiperparámetros

### Curva de validación de K-NN
`results/figures/knn_curva_validacion_k.png` (10 folds, datos escalados). En el sintético, la accuracy máxima es 0.8622 con pesos uniformes y 0.8667 con pesos por distancia, ambos en k = 9; con k = 1 es 0.8122. En Wine, el máximo es 0.9778 (uniformes, k = 15; por distancia, k = 21), frente a 0.9552 con k = 1. En el sintético, aumentar k sobre ≈ 9 reduce ligeramente la accuracy (0.8433 con k = 21 con pesos uniformes).

### Default vs. ajustado (CV anidada: interna 5, externa 10)

| dataset   | modelo   | acc_defecto     | acc_ajustado    |   mejora | folds_mejor/igual/peor   |   p_wilcoxon | significativa_(p<0.05)   |
|:----------|:---------|:----------------|:----------------|---------:|:-------------------------|-------------:|:-------------------------|
| sintetico | K-NN     | 0.8533 ± 0.0362 | 0.8589 ± 0.0348 |   0.0056 | 6/1/3                    |       0.3438 | no                       |
| sintetico | SVM      | 0.8600 ± 0.0410 | 0.8600 ± 0.0378 |  -0      | 4/3/3                    |       0.9375 | no                       |
| sintetico | MLP      | 0.8567 ± 0.0361 | 0.8556 ± 0.0367 |  -0.0011 | 3/4/3                    |       0.9375 | no                       |
| wine      | K-NN     | 0.9608 ± 0.0271 | 0.9608 ± 0.0377 |   0      | 1/8/1                    |       1      | no                       |
| wine      | SVM      | 0.9830 ± 0.0274 | 0.9886 ± 0.0241 |   0.0056 | 1/9/0                    |       1      | no                       |
| wine      | MLP      | 0.9778 ± 0.0287 | 0.9833 ± 0.0268 |   0.0056 | 1/9/0                    |       1      | no                       |

**¿Mejoran los resultados?** Numéricamente, la mejora media es de a lo sumo 0.0056 (K-NN en el sintético, SVM y MLP en Wine) y en tres casos es ≤ 0 (SVM y MLP en el sintético, K-NN en Wine). **Ninguna diferencia es estadísticamente significativa**: los p-valores de Wilcoxon van de 0.34 a 1.0. Con 10 folds, la prueba tiene poca potencia, y los folds comparten datos, por lo que los p-valores son orientativos. La conclusión honesta es que, con estos datos, los parámetros por defecto ya son casi tan buenos como los ajustados. En el sintético, el límite lo impone el ruido de etiquetas (10 % volteado), no los hiperparámetros.

**¿Cómo encontrar los mejores parámetros para cada dataset?** Con validación cruzada: una búsqueda (rejilla si el espacio es pequeño, aleatoria con distribuciones log-uniformes si es grande o continuo) que evalúe cada configuración con CV sobre el conjunto de entrenamiento; y, para **estimar** el desempeño real, anidarla en una CV externa, de modo que el test nunca influya en la elección. Los parámetros recomendados, buscados con CV de 5 folds sobre todo el dataset, son:

| dataset   | modelo   | mejores_parametros                                                                                                                                      |   acc_cv_5_folds_(optimista) |
|:----------|:---------|:--------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------:|
| sintetico | K-NN     | {'k': 11, 'metric': 'manhattan', 'weights': 'distance'}                                                                                                 |                       0.8622 |
| sintetico | SVM      | {'C': 2.3336742079960358, 'gamma': 1.0105405843110133, 'kernel': 'rbf'}                                                                                 |                       0.8633 |
| sintetico | MLP      | {'activation': 'tanh', 'alpha': 0.001304907355036239, 'early_stopping': True, 'hidden_layer_sizes': (50,), 'learning_rate_init': 0.05981221901152557}   |                       0.8667 |
| wine      | K-NN     | {'k': 21, 'metric': 'euclidean', 'weights': 'uniform'}                                                                                                  |                       0.9832 |
| wine      | SVM      | {'C': 0.06405419674802731, 'gamma': 0.1843069139534015, 'kernel': 'linear'}                                                                             |                       0.9887 |
| wine      | MLP      | {'activation': 'tanh', 'alpha': 0.0036281404040243792, 'early_stopping': False, 'hidden_layer_sizes': (50,), 'learning_rate_init': 0.06823493012435795} |                       0.9832 |

La última columna es optimista (se elige el máximo entre muchas configuraciones sobre los mismos datos), por lo que no sustituye al desempeño de la tabla anterior. Además, los mejores parámetros **no son estables** entre folds externos: p. ej., el *k* elegido por K-NN fue de 5 a 13 en el sintético (con pesos por distancia en los 10 folds) y de 1 a 9 en Wine (con Manhattan en los 10 folds). Esto refleja que muchas configuraciones rinden casi igual. En promedio por fold externo, la búsqueda tardó ≈ 8.1 s (MLP) y ≈ 3.0 s (SVM) en el sintético, y ≈ 0.3 a 0.4 s para K-NN.

## 7. Discusión

**Robustez ante el ruido estructurado.** En el sintético, ningún modelo supera ≈ 0.87, un techo consistente con el 10 % de etiquetas volteadas (las etiquetas observadas son erróneas por construcción) y con las clases solapadas. Los modelos que suavizan (SVM, MLP, K-NN con k > 1) quedan por encima de los que memorizan (1-NN) o suponen una forma rígida (Distancia Mínima), aunque la mayor brecha (1-NN frente a K-NN, ≈ 0.04 a 0.05 de accuracy) es comparable a la desviación estándar en K-Fold (≈ 0.04), por lo que no se puede afirmar con rotundidad.

**Por qué 1-NN sufre con etiquetas ruidosas.** 1-NN toma la etiqueta de un único vecino: si ese vecino es uno de los puntos volteados, la predicción es incorrecta, y además el modelo se ajusta perfectamente al entrenamiento (accuracy de entrenamiento = 1.000), incluyendo el ruido, por lo que traza "islas" alrededor de cada etiqueta volteada (figura de fronteras). Como los volteos están concentrados en las fronteras, justo ahí hay más puntos erróneos. K-NN con k > 1 promedia entre varios vecinos y diluye esos errores, lo que coincide con la subida de 0.8122 (k = 1) a 0.8533 (k = 5) y 0.8622 (k = 9). Los outliers, en cambio, están lejos de las clases principales; es esperable que afecten poco a los modelos locales, aunque no se midió su efecto por separado.

**Efecto del escalado.** En Wine el escalado es determinante para los modelos que dependen de distancias o de márgenes: SVM pasa de 0.6389 a 0.9852 (Hold-Out), K-NN de 0.687 a 0.95, Distancia Mínima de 0.70 a 0.963 y 1-NN de 0.7463 a 0.9574; MLP mejora menos (0.9 a 0.9722) y Naive Bayes no cambia (0.9741 en ambos casos), porque modela cada rasgo por separado y es invariante a escala. Esto ocurre porque los rasgos de Wine tienen escalas muy distintas, y los de mayor magnitud dominan la distancia. En el sintético los rasgos ya tienen escalas parecidas y el efecto es pequeño (≤ 0.01 salvo en 1-NN); en 1-NN el escalado incluso reduce la accuracy (0.8352 sin escalar frente a 0.8067 escalado en Hold-Out), posiblemente porque el escalado da el mismo peso a los 3 rasgos de ruido que a los informativos (hipótesis, no verificada con un experimento aparte).

**Compromiso entre costo y precisión.** En Wine, Naive Bayes y Distancia Mínima (≈ 0.001 s de fit y predict) logran 0.9719 con K-Fold, frente a 0.9830 de SVM, es decir, ≈ 0.011 menos de accuracy por un costo mucho menor, aunque dentro de las desviaciones. En el sintético, MLP y SVM obtienen la mejor accuracy, pero MLP entrena más de 200 veces más lento que Distancia Mínima o Naive Bayes (0.34 a 0.42 s frente a 0.001 a 0.002 s), para una ganancia con K-Fold de ≈ 0.013 (frente a Naive Bayes) a ≈ 0.027 (frente a Distancia Mínima). Los métodos basados en vecinos escalan mal en predicción con *n* (≈ 2 s para 1000 muestras con n = 20 000), y el MLP escala mal en entrenamiento. El ajuste de hiperparámetros añade un costo considerable (por ejemplo, ≈ 8 s por búsqueda en MLP) y aquí no produjo mejoras significativas.

**Limitaciones.** Las desviaciones estándar entre particiones son comparables a varias diferencias entre modelos; los tiempos dependen de la máquina; el sintético se generó con una única semilla; y los conjuntos son pequeños.

## 8. Conclusiones

1. Con parámetros por defecto, SVM obtuvo la mayor accuracy media en ambos datasets y en ambas validaciones (sintético: 0.8615 y 0.8600; Wine: 0.9852 y 0.9830), pero las diferencias con MLP y, en Wine, con Naive Bayes y Distancia Mínima caen dentro de la variabilidad.
2. 1-NN fue el más afectado por el ruido de etiquetas del sintético (0.8122 con K-Fold); K-NN con k = 5 lo mejora (0.8533), y la curva de validación indica k ≈ 9.
3. El escalado es crítico en Wine para todos los modelos basados en distancias y en SVM, e irrelevante para Naive Bayes.
4. El costo de predicción de K-NN crece linealmente con *n* (pendiente ≈ 1.1) y el de entrenamiento de SVM crece con pendiente ≈ 1.8 a 1.9, de acuerdo con la teoría.
5. El ajuste de hiperparámetros con CV anidada no produjo mejoras estadísticamente significativas (p ≥ 0.34 en los 6 casos).

## 9. Referencias

- Cover, T. y Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27.
- Cortes, C. y Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20, 273-297.
- Rumelhart, D., Hinton, G. y Williams, R. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533-536.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825-2830.
- Harris, C. R. et al. (2020). Array programming with NumPy. *Nature*, 585, 357-362.
- Bergstra, J. y Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR*, 13, 281-305.
- Cawley, G. y Talbot, N. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR*, 11, 2079-2107.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*, 7, 1-30.
- Frénay, B. y Verleysen, M. (2014). Classification in the presence of label noise: a survey. *IEEE TNNLS*, 25(5), 845-869.
- Forina, M. et al. Wine dataset, UCI Machine Learning Repository.

## 10. Anexos

- Repositorio: **https://github.com/TheBizZ-Coder/practica3-clasificador**
- Conversación con la IA utilizada: carpeta `ia_conversacion/`.
- Reproducción completa: `python experiments/run_all.py` (semilla global 0 por defecto).
- Tablas completas por partición: `results/tables/defaults_por_particion.csv`, `tuning_folds.csv`, `complejidad_empirica.csv`.
