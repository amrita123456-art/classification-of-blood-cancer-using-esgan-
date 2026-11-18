import os
from flask import Flask, send_from_directory
from config import SECRET_KEY, UPLOAD_FOLDER, REPORTS_FOLDER, BASE_DIR
from database.db import init_db
from routes.main import main_bp
from routes.analysis import analysis_bp
from routes.history import history_bp
from routes.metrics import metrics_bp
from routes.auth import auth_bp

def create_app():
    """Application factory for BloodAI Flask application."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

    # Initialize SQLite database
    init_db()

    # Register Route Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(metrics_bp)

    # Route to serve evaluation report charts
    @app.route("/reports/<path:filename>")
    def serve_reports(filename):
        return send_from_directory(str(REPORTS_FOLDER), filename)

    return app

if __name__ == "__main__":
    app = create_app()
    print("\n" + "=" * 65)
    print(" BloodAI: ESRGAN + MobileNetV2 Leukemia Classification")
    print(" Clinical AI Dashboard started on http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
