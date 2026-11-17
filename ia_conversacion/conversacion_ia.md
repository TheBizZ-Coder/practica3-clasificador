# Conversación con la IA: lluvia de ideas sobre los 6 clasificadores

**Usuario:** Hola, tengo una práctica donde debo comparar seis clasificadores: Distancia Mínima, 1-NN, K-NN, Naive Bayes, SVM y MLP. Los voy a probar en un dataset sintético con ruido y en el dataset Wine. ¿Me ayudas con ideas para abordarla? Todavía no quiero código, solo entender qué esperar de cada uno.

**IA:** Claro. Una buena forma de empezar es pensar en qué "suposición" hace cada clasificador sobre los datos. Si entiendes eso, casi puedes predecir cuál va a sufrir con el ruido, cuál con la escala y cuál va a ser más lento. Vamos uno por uno, y al final armamos un plan de comparación.

**Usuario:** Perfecto. Empieza por el más simple.

**IA:** El más simple es **Distancia Mínima**. Resume cada clase con un solo punto, su centroide (el promedio de sus muestras), y un punto nuevo se asigna a la clase cuyo centroide quede más cerca. Es muy barato: entrenar es sacar promedios y predecir es comparar contra unos pocos centroides. Su debilidad es que es muy rígido: produce fronteras rectas y asume que cada clase es una nube compacta alrededor de su promedio. Si una clase está muy dispersa o tiene forma rara, se equivoca. También depende de la escala, porque usa distancias. Su gran ventaja es que, al promediar, no se deja engañar fácilmente por unas pocas etiquetas erróneas.

**Usuario:** Y 1-NN, ¿qué tiene de especial?

**IA:** **1-NN** va al extremo contrario: no resume nada, guarda todos los datos de entrenamiento y le pone al punto nuevo la etiqueta de su vecino más cercano. Eso le da fronteras muy flexibles y se ajusta perfecto al entrenamiento. El problema es justo ese: si una etiqueta del entrenamiento está equivocada, 1-NN la copia fielmente. Con ruido en las etiquetas cerca de las fronteras, es de los que más sufre, porque cada etiqueta errónea crea una pequeña "isla" donde predice mal. Además, entrenar no cuesta nada, pero predecir sí, porque debe comparar contra todo el conjunto guardado. Y también es sensible a la escala de los rasgos.

**Usuario:** Entonces K-NN es una versión mejorada de 1-NN, ¿no?

**IA:** Se puede ver así. **K-NN** en lugar de fiarse de un solo vecino, consulta a k vecinos y deja que voten. Al promediar varias opiniones, los errores aislados se diluyen, así que suele ser más robusto al ruido de etiquetas. El parámetro clave es k: con k muy pequeño se parece a 1-NN (fronteras irregulares, mucha varianza); con k muy grande se vuelve demasiado suave y empieza a ignorar detalles reales. También puedes decidir si todos los vecinos pesan igual o si los más cercanos pesan más, y qué distancia usar. Mantiene el mismo costo de predicción que 1-NN, que crece con la cantidad de datos de entrenamiento.

**Usuario:** Sigamos con Naive Bayes.

**IA:** **Naive Bayes gaussiano** es un clasificador probabilístico. Para cada clase estima, rasgo por rasgo, una campana de Gauss (media y varianza), y luego combina esa información usando el teorema de Bayes, asumiendo que los rasgos son independientes entre sí dentro de cada clase. Esa suposición casi nunca es del todo cierta, de ahí lo de "ingenuo", pero en la práctica funciona sorprendentemente bien y es rapidísimo. Algo interesante para tu práctica: como modela cada rasgo por separado, la escala le da igual. Si estandarizas o no los datos, su resultado casi no cambia. Su punto débil aparece cuando los rasgos están muy correlacionados o cuando las clases no se parecen a una campana.

**Usuario:** Ahora SVM.

**IA:** **SVM** busca la frontera que deja el mayor margen posible entre las clases. Con un kernel, como el RBF, puede curvar esa frontera para separar datos que no son lineales. Tiene dos botones principales: C, que decide cuánto castigar los errores (valores altos buscan ajustarse más a los datos, valores bajos toleran más errores y suavizan), y gamma, que controla qué tan "local" es el kernel. Es muy bueno en general, pero entrenar es más caro que en los modelos anteriores y el costo crece rápido con la cantidad de datos. Y ojo con la escala: como el kernel usa distancias, si los rasgos tienen magnitudes muy distintas puede fallar muchísimo si no los estandarizas.

**Usuario:** Y el último, MLP.

**IA:** El **MLP** es una red neuronal con una o varias capas ocultas. Aprende combinaciones de los rasgos ajustando pesos poco a poco, con muchas pasadas sobre los datos. Es el más flexible de los seis, pero también el más sensible a decisiones de diseño: cuántas capas y neuronas, qué función de activación, cuánta regularización, cuántas iteraciones. Es el más lento de entrenar, y a veces avisa que no terminó de converger si le das pocas iteraciones. Con datos pequeños puede sobreajustar si no lo controlas. También agradece que los rasgos estén en escalas parecidas, aunque es menos dramático que en SVM o K-NN.

**Usuario:** Ya con eso, ¿cómo organizo la comparación? ¿Qué debería esperar?

**IA:** Aquí van algunas ideas para organizarte, y de paso te dejo hipótesis que luego puedes confirmar o refutar:

1. **Ruido en etiquetas.** Espero que 1-NN sea el más afectado, porque memoriza el ruido. K-NN con k mayor a uno debería mejorar, y los modelos que suavizan, como SVM y MLP, deberían manejarlo bien. Distancia Mínima y Naive Bayes, al promediar, deberían quedar en un punto intermedio.
2. **Escala de los rasgos.** Los que dependen de distancias o márgenes (Distancia Mínima, 1-NN, K-NN, SVM) deberían depender mucho de estandarizar. Naive Bayes casi no. El MLP, algo en medio. Wine es perfecto para ver esto porque sus rasgos tienen magnitudes muy distintas.
3. **Costo.** Distancia Mínima y Naive Bayes deberían ser los más baratos. K-NN y 1-NN deberían ser baratos al entrenar pero lentos al predecir. SVM y sobre todo MLP deberían ser los más caros al entrenar.
4. **Rasgos inútiles.** Si el dataset sintético incluye columnas de ruido sin información, los métodos basados en distancia se ven más perjudicados, porque esas columnas también cuentan al medir qué tan cerca están dos puntos.
5. **Datos pequeños.** Wine tiene pocas muestras, así que las diferencias entre modelos pueden ser pequeñas y quedar dentro de la variación normal. Conviene repetir con varias particiones y no fiarse de una sola corrida.
6. **Ajuste de hiperparámetros.** Antes de afinar, vale la pena preguntarse si el modelo con valores por defecto ya está cerca de su techo. Si el ruido de las etiquetas es lo que limita el desempeño, ajustar parámetros quizá no cambie mucho.

**Usuario:** Me gusta. ¿Algo más a tener en cuenta?

**IA:** Dos cosas. Primero, justicia en la comparación: todos los modelos deben ver las mismas particiones de entrenamiento y prueba, y el escalado debe ajustarse solo con los datos de entrenamiento para no "espiar" el conjunto de prueba. Segundo, honestidad al interpretar: si un modelo gana por una diferencia menor que la variación entre particiones, lo correcto es decir que no hay una ventaja clara. Con esas dos ideas, la práctica queda bien planteada, y el análisis se vuelve más interesante porque cada resultado se puede explicar con la suposición que hace cada clasificador.
