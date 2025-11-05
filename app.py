from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Load saved model
model_path = Path("artifacts") / "model.pkl"
model_data = joblib.load(model_path)
model = model_data["model"]
FEATURES = model_data["features"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form values in the correct order
    values = [float(request.form.get(f)) for f in FEATURES]
    arr = np.array(values).reshape(1, -1)

    # Make prediction
    pred = int(model.predict(arr)[0])
    proba = round(float(model.predict_proba(arr)[0, 1]), 3)

    return render_template("index.html", pred=pred, proba=proba)

if __name__ == "__main__":
    app.run(debug=True)