Mi ejemplo es una aplicación de análisis de sentimientos implementada en Python. Esta app utiliza la librería pysentimiento, la cual internamente carga un modelo basado en la arquitectura Transformer, específicamente una variante de BERT llamada RoBERTuito.

RoBERTuito fue preentrenado con más de 500 millones de tweets en español y luego ajustado (fine-tuned) con un corpus etiquetado en 2020 (TASS) para tareas sociales como el análisis de sentimientos. Gracias a este entrenamiento masivo, el modelo ha aprendido a comprender tanto expresiones coloquiales como estructuras lingüísticas formales propias del español.

Este modelo interpreta el significado de las palabras a través de un mecanismo llamado autoatención (self-attention), que le permite analizar cómo se relaciona cada palabra con todas las demás, sin importar su distancia en la frase. Esta atención se aplica bidireccionalmente, es decir, cada palabra se representa considerando tanto lo que viene antes como lo que viene después, a diferencia de modelos más antiguos que solo miraban hacia una dirección.

Aunque RoBERTuito no usa literalmente un "doble encoder", su diseño bidireccional le permite construir una representación contextual rica que incluye tanto información sintáctica (la estructura de la oración) como semántica (el significado profundo de las palabras). Esto se logra a través de una pila de 12 capas Transformer, cada una compuesta por mecanismos de atención multi-cabeza y redes neuronales densas (feedforward), que refinan progresivamente la comprensión del texto.

Finalmente, una capa de clasificación transforma la representación del texto en una distribución de probabilidad sobre tres clases: positivo, negativo o neutral, mediante una función softmax, que indica qué sentimiento predomina y con cuánta certeza.

Gracias a esta arquitectura profunda y contextual, la aplicación es capaz de interpretar frases en lenguaje natural (formales o informales) y clasificarlas automáticamente según el sentimiento que expresan, siendo especialmente eficaz en textos breves como los de redes sociales.

![Example](https://github.com/RickUbb/feelingAnalysis/blob/main/Example.jpeg?raw=true)
