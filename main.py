from flask import Flask
from flask_cors import CORS
# from api.routes import routes # ¡Comenta esta línea temporalmente!

app = Flask(__name__)
CORS(app)
# app.register_blueprint(routes) # ¡Comenta esta línea temporalmente!

@app.route("/", methods=["GET"]) # Agrega una ruta simple
def home():
    return "¡Hola desde Flask en Azure!"

if __name__ == "__main__":
    app.run(debug=True, port=5000)