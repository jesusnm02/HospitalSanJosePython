import pickle
from pathlib import Path
import subprocess

model_path = Path(__file__).parent / "model.pkl"

if not model_path.exists():
    print("🔄 Modelo no encontrado. Entrenando modelo...")
    subprocess.run(["python", "train/train_model.py"], check=True)

with open(model_path, "rb") as f:
    vectorizer, model, df = pickle.load(f)

respuestas_generales = {
    "hola": "¡Hola! ¿En qué puedo ayudarte con tus síntomas?",
    "buenos días": "¡Buenos días! Estoy lista para ayudarte.",
    "cómo estás": "Estoy perfecto. ¿Qué síntomas deseas analizar?",
    "gracias": "¡De nada! Si tienes más preguntas médicas, aquí estoy.",
    "adiós": "¡Cuídate mucho! Hasta luego."
}


def predecir_medicamento(sintomas: str) -> str:
    texto = sintomas.lower().strip()

    # Respuesta general si es un saludo o mensaje informal
    for clave, respuesta in respuestas_generales.items():
        if clave in texto:
            return respuesta

    # Predicción del modelo
    try:
        X = vectorizer.transform([texto])
        pred = model.predict(X)[0]

        # Buscar información relacionada al medicamento
        fila = df[df["medicamentos"] == pred]
        if fila.empty:
            return ("He detectado una posible coincidencia, pero no tengo información suficiente "
                    "para darte una recomendación clara. Por favor consulta con un médico.")

        fila = fila.iloc[0]
        descripcion = fila.get("description", "sin descripción")
        tratamiento = fila.get("tratamiento", "tratamiento no especificado")

        return (f"Según los síntomas indicados, se sospecha: **{descripcion}**. "
                f"Se recomienda el tratamiento: **{tratamiento}** y administrar **{pred}**.")

    except Exception as e:
        print(f"[Error IA] {e}")
        return ("Ocurrió un error al procesar tu solicitud. Por favor intenta nuevamente o consulta con un médico.")
