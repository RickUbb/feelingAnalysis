from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Cargar el tokenizador y el modelo BETO ajustado para análisis de sentimientos
tokenizer = AutoTokenizer.from_pretrained("finiteautomata/beto-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/beto-sentiment-analysis")

# Función para analizar el sentimiento de un texto
def analiza(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    etiquetas = ['NEG', 'NEU', 'POS']
    prediccion = etiquetas[probs.argmax()]
    return {
        "texto": texto,
        "sentimiento": prediccion,
        "probabilidades": {etiquetas[i]: float(probs[0][i]) for i in range(len(etiquetas))}
    }

# Ejemplos de uso
print(analiza("Me encanta la idea de hacer un remake de Karate Kid"))
print(analiza("A ver cuándo por fin termina la serie de Cobra Kai de una vez"))
print(analiza("Estos tíos de Netflix van a quemar una idea brutal, estoy feliz"))
