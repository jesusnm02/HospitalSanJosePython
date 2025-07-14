import pickle
from pathlib import Path

model_path = Path(__file__).parent / "model.pkl"

try:
    with open(model_path, "rb") as f:
        vectorizer, model, df = pickle.load(f)
    print("✅ Modelo cargado correctamente.")
except FileNotFoundError:
    print(f"❌ ERROR: El archivo del modelo no se encontró en: {model_path}")
    print("Asegúrate de que 'model.pkl' esté entrenado y desplegado con la aplicación.")
    raise RuntimeError("El modelo predictivo 'model.pkl' no se encontró al iniciar la aplicación.")
except Exception as e:
    print(f"❌ ERROR al cargar el modelo: {e}")
    raise

respuestas_generales = {
    "hola": "¡Hola! ¿En qué puedo ayudarte con tus síntomas?",
    "buenos días": "¡Buenos días! Estoy lista para ayudarte.",
    "cómo estás": "Estoy perfecto. ¿Qué síntomas deseas analizar?",
    "gracias": "¡De nada! Si tienes más preguntas médicas, aquí estoy.",
    "adiós": "¡Cuídate mucho! Hasta luego."
}

def predecir_medicamento(sintomas: str) -> str:
    texto = sintomas.lower().strip()

    for clave, respuesta in respuestas_generales.items():
        if clave in texto:
            return respuesta

    try:
        X = vectorizer.transform([texto])
        pred = model.predict(X)[0]

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