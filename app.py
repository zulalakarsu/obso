import warnings
warnings.filterwarnings("ignore", 
    message="Found unknown categories in columns.*during transform")

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ─── CONFIG ───────────────────────────────────────────────────────────
MODELS_DIR = "models"

# ─── LOAD THE NEW PIPELINES ────────────────────────────────────────────
models = {
    "boston":   joblib.load(os.path.join(MODELS_DIR, "xgboost_Boston.pkl")),
    "new_york": joblib.load(os.path.join(MODELS_DIR, "xgboost_New_York.pkl"))
}

# ─── FEATURE LIST ──────────────────────────────────────────────────────
NUM_FEATURES = [
    'Building_Size','Floorplate','Building Age','Renovated','Building Class Score',
    'Commuter Rail Score','Subway Service Score','Highway Proximity Score','Parking Type Score',
    'Energy_Efficiency Score','Energy-Star','occupancy rate','assessed_value','OpEx','Owner Occu Score'
]
CAT_FEATURES = ['Submarket']
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# ─── FLASK APP ─────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "models": list(models.keys())
    })

@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        data = request.get_json(force=True)
    else:
        if request.form:
            data = request.form.to_dict()
        else:
            data = request.get_json(force=True)

    city = data.get("city", "").lower().replace(" ", "_")
    if city not in models:
        return jsonify({
            "error": "invalid_city",
            "allowed_cities": list(models.keys())
        }), 400

    # Convert numeric fields from string to appropriate type
    for f in NUM_FEATURES:
        if f in data:
            try:
                data[f] = float(data[f]) if "." in str(data[f]) else int(data[f])
            except Exception:
                pass
    df = pd.DataFrame([data])
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        return jsonify({
            "error": "missing_features",
            "fields": missing
        }), 400

    X_raw = df[ALL_FEATURES]
    pipe  = models[city]

    proba = pipe.predict_proba(X_raw)[0].tolist()

    # The predicted score should be the class with the highest probability (1-based)
    pred = int(proba.index(max(proba))) + 1

    result = {
        "city": city,
        "prediction": pred,
        "probabilities": {str(i+1): round(p, 4) for i, p in enumerate(proba)}
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5002)))
