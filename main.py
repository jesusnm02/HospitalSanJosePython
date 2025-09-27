import os
from flask import Flask
from flask_cors import CORS
from api.routes import routes

app = Flask(__name__)
CORS(app)
app.register_blueprint(routes)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # usa el puerto de Azure
    app.run(host="0.0.0.0", port=port, debug=True)
