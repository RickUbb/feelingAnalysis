# Importamos el tokenizador y el modelo de clasificación desde la librería Transformers de Hugging Face.
# Estos elementos permiten convertir texto en vectores y usar un modelo entrenado para clasificar sentimiento.
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Importamos PyTorch, la librería de aprendizaje profundo que ejecuta el modelo.
import torch

# Importamos funciones de activación desde torch.nn.functional.
# Aquí usaremos softmax para convertir los resultados del modelo en probabilidades.
import torch.nn.functional as F

# -----------------------
# CONFIGURACIÓN DEL MODELO
# -----------------------

# Indicamos el nombre del modelo a utilizar desde Hugging Face.
# Este es un modelo RoBERTuito ajustado para análisis de sentimientos en español.
modelo = "pysentimiento/robertuito-sentiment-analysis"

# Cargamos el tokenizador correspondiente al modelo. Convierte texto en IDs numéricos entendibles por el modelo.
tokenizer = AutoTokenizer.from_pretrained(modelo)

# Cargamos el modelo preentrenado como clasificador de secuencias.
# Este modelo ha sido entrenado para etiquetar frases como POS (positivo), NEG (negativo), o NEU (neutral).
model = AutoModelForSequenceClassification.from_pretrained(modelo)

# -----------------------
# FUNCIÓN DE ANÁLISIS
# -----------------------

# Definimos una función llamada 'analiza' que recibe un texto en español y devuelve su sentimiento.
def analiza(texto):
    # Convertimos el texto a tensores (vectores) con el tokenizador.
    # - return_tensors="pt" lo convierte a tensores de PyTorch.
    # - truncation=True recorta el texto si es muy largo.
    # - padding=True añade ceros si es más corto que el máximo.
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    
    # Desactivamos el cálculo del gradiente porque no vamos a entrenar el modelo, solo usarlo.
    with torch.no_grad():
        # Pasamos los vectores al modelo y obtenemos los logits (valores sin procesar).
        logits = model(**inputs).logits
    
    # Aplicamos softmax para convertir los logits en probabilidades entre 0 y 1.
    probs = F.softmax(logits, dim=1)
    
    # Definimos las etiquetas que el modelo puede predecir.
    etiquetas = ['NEG', 'NEU', 'POS']
    
    # Seleccionamos la etiqueta con mayor probabilidad como predicción final.
    prediccion = etiquetas[probs.argmax()]
    
    # Retornamos un diccionario con:
    # - el texto original
    # - el sentimiento predicho
    # - las probabilidades detalladas para cada clase
    return {
        "texto": texto,
        "sentimiento": prediccion,
        "probabilidades": {etiquetas[i]: float(probs[0][i]) for i in range(len(etiquetas))}
    }

# -----------------------
# PRUEBAS DEL MODELO
# -----------------------

# Analizamos tres frases distintas y mostramos el resultado de cada una.

# Frase con sentimiento positivo
print(analiza("Me encanta la idea de hacer un remake de Karate Kid"))

# Frase con tono posiblemente neutral o ambivalente
print(analiza("A ver cuándo por fin termina la serie de Cobra Kai de una vez"))

# Frase con frustración y crítica (probablemente negativa)
print(analiza("Estos tíos de Netflix van a quemar una idea brutal"))
