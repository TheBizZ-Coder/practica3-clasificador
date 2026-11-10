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