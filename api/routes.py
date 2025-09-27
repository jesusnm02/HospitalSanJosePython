from flask import Blueprint, request, jsonify
from app.predictor import predecir_medicamento

routes = Blueprint("routes", __name__)

@routes.route("/ia/sugerir", methods=["POST"])
def sugerir():
    data = request.get_json() or {}
    sintomas = data.get("sintomas", "")
    respuesta = predecir_medicamento(sintomas)
    return jsonify({"respuesta": respuesta})
