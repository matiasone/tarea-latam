* Lo primero que se hizo considerando hasta la etapa 4 del archivo .ipynb fue, para la librería sns.barplot, fue definir x=dataframe.index e y=dataframe.values. De esta manera, asignando 'x' e 'y' se evita el error que tiraba al correr el código, referente a que sns.barplot solo toma al argumento 'data', ya que al usar 'x' e 'y' deja de ser necesario dicho argumento.

Para elegir el modelo mas adecuado, primero observamos la matriz de confusión, la cual entrega información sobre verdaderos positivos (1 y 1), falsos positivos (1 y 0), verdaderos negativos (0 y 0) y falsos negativos (0 y 1). En este caso, positivo es que hay delay y negativo es que no hay delay.

Las métricas a evaluar son las siguientes:

accuracy = (VP + VN) / (VP + VN + FP + FN)

Accuracy es la típica métrica para evaluar modelos de machine learning, pero esta no es tan confiable para modelos con un conjunto de datos no balanceado o en dominios específicos. 

recall = VP/(VP + FN)

precision = VP/(VP + FP)

F1-score = 2 * Precision * Recall/(Precision + Recall)

En este caso específico de predicción de atraso en un vuelo, pensando en la satisfacción del cliente, a parte de buscar la mayor cantidad de aciertos (VP + VN), se debe priorizar que hayan más Falsos Positivos en vez de Falsos Negativos, ya que es mejor avisar de un probable atraso que no avisar de un vuelo que se atrasará. Por lo tanto, debemos priorizar un modelo que tenga un mayor índice sensibilidad (recall) que de precisión.

Para plasmar esta prioridad en una métrica, podemos usar la métrica F-beta-score:

F-beta-score = F1-score * (1 + beta^2)/(2*beta^2). [Fernández et al., 2018] recomienda un beta = 2 para darle más peso al recall. Entonces, agregamos en el código print("f2-score:",fbeta_score(y_test, y_pred, beta=2)), con lo que obtenemos el valor F2-score para cada modelo. 

(1) XGBoost: 
    accuracy = 0.82, f2score = 0.027375485694101028
(2) Logistic Regression:
    accuracy = 0.81, f2score = 0.03745976002341235
(3) XGBoost with Feature Importance and with Balance:
    accuracy = 0.55, f2score = 0.5090366731005439
(4) XGBoost with Feature Importance but without Balance:
    accuracy = 0.81, f2score = 0.00887679015268079
(5) Logistic Regression with Feature Importante and with Balance:
    accuracy = 0.55, f2score = 0.5076497566782201
(6) Logistic Regression with Feature Importante but without Balance:
    accuracy = 0.81, f2score = 0.01592168887840547

El modelo con mejor F2-score fue el (5), por lo que nos quedamos con ese.

Posteriormente, se completa la clase DelayModel() con lo pedido en el enunciado, utilizando lo mostrado en el jupyter notebook para el modelo de Logistic Regression con selección de características y balance.
Se crea una instancia de la clase y se importa dicha instancia para hacer las consultas al modelo a través de la api.

En la parte de la API, se tratan los casos mostrados en test_api donde se debe retornar un error 400, se completa el diccionario que será utilizado como input para el modelo con la información entregada y la ausencia de esta, así completando los valores para cada una de las 10 características seleccionadas, las cuales son las siguientes:

top_10_features = [
    "OPERA_Latin American Wings", 
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air"
]

Así, si se encuentra uno de esas características en el input, se completa como verdadero, mientras que si no se encuentra, se completa como falso. Finalmente se realiza la consulta al modelo.

Para consultar al modelo, se cambió en test la forma de consultar, pasando de 'json=data' a simplemente 'data' para entregar el input. No se pudo probar la ejecución del código debido al error "ERROR tests/api/test_api.py - AttributeError: module 'anyio' has no attribute 'start_blocking_portal'"

No pude realizar la parte III y parte IV por falta de tiempo. Gracias por la oportunidad!






