import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load your trained XGBoost model saved as file_XGBoost.pkl (in the same folder as app.py)
with open("file_XGBoost.pkl", "rb") as f:
    model = pickle.load(f)

# Feature order must match training
FEATURES = [
    "Clump_thickness",
    "Uniformity_of_cell_size",
    "Uniformity_of_cell_shape",
    "Marginal_adhesion",
    "Single_epithelial_cell_size",
    "Bare_nuclei",
    "Bland_chromatin",
    "Normal_nucleoli",
    "Mitoses",
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read inputs from the form in the same order as FEATURES
        vals = [float(request.form[name]) for name in FEATURES]
        X = np.array([vals])  # shape (1, 9)

        pred = model.predict(X)  # expects 0/1 from your training
        label = "Malignant" if int(pred[0]) == 1 else "Benign"

        return render_template("index.html", predict=label)
    except Exception as e:
        return render_template("index.html", predict=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
