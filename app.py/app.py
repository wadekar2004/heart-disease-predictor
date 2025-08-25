from flask import Flask, render_template, request
import os, pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# ------------------------------
# Load your trained model
# ------------------------------
MODEL_PATHS = [
    "heart_model.pkl",
    os.path.join(os.path.dirname(__file__), "heart_model.pkl"),
    r"C:\Users\Aditya\Desktop\ML\heart-disease-project\heart_model.pkl"
]

model = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        with open(p, "rb") as f:
            model = pickle.load(f)
        print(f"[INFO] Loaded model from: {p}")
        break

if model is None:
    raise RuntimeError("Could not load heart_model.pkl from known paths.")

# ------------------------------
# Training feature order (UCI Heart)
# ------------------------------
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# ------------------------------
# Helpers
# ------------------------------
def _df(d):
    return pd.DataFrame([[d[c] for c in FEATURES]], columns=FEATURES)

def _clamp(d):
    d["age"]      = float(np.clip(d["age"], 1, 120))
    d["trestbps"] = float(np.clip(d["trestbps"], 80, 250))
    d["chol"]     = float(np.clip(d["chol"], 100, 600))
    d["thalach"]  = float(np.clip(d["thalach"], 60, 220))
    d["oldpeak"]  = float(np.clip(d["oldpeak"], 0.0, 10.0))
    d["ca"]       = float(np.clip(d["ca"], 0, 4))
    return d

def _float(v, default):
    try:
        return float(v)
    except:
        return float(default)

# ------------------------------
# Map Thal field (handles 1-3 vs 3-6-7 coding)
# ------------------------------
def map_thal(user_thal_value, thal_mode):
    v = int(_float(user_thal_value, 0))
    if thal_mode == "367":
        if v == 1: return 3
        if v == 2: return 6
        if v == 3: return 7
        return 0
    return v

def predict_proba_vec(Xdf):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xdf)[0]
        classes = getattr(model, "classes_", np.array([0, 1]))
        return proba, classes
    # No predict_proba → emulate
    y = model.predict(Xdf)[0]
    classes = getattr(model, "classes_", np.array([y]))
    proba = np.zeros(len(classes))
    for i, c in enumerate(classes):
        if c == y:
            proba[i] = 1.0
    return proba, classes

def idx_of(classes, label):
    where = np.where(classes == label)[0]
    return int(where[0]) if len(where) else None

def best_disease_label(classes, p_healthy, p_sick):
    best_i, best_gain = 0, -1e9
    for i in range(len(classes)):
        gain = p_sick[i] - p_healthy[i]
        if gain > best_gain:
            best_gain, best_i = gain, i
    return classes[best_i]

# ------------------------------
# Calibrate disease label + THAL mode
# ------------------------------
def calibrate_thal_and_label():
    healthy = {
        "age": 25, "sex": 0, "cp": 0, "trestbps": 110, "chol": 180,
        "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0,
        "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 1
    }
    sick = {
        "age": 68, "sex": 1, "cp": 3, "trestbps": 170, "chol": 290,
        "fbs": 1, "restecg": 2, "thalach": 100, "exang": 1,
        "oldpeak": 3.5, "slope": 2, "ca": 3, "thal": 3
    }

    results = []
    for mode in ["134", "367"]:
        H = healthy.copy(); S = sick.copy()
        H["thal"] = map_thal(H["thal"], mode)
        S["thal"] = map_thal(S["thal"], mode)

        pH, cH = predict_proba_vec(_df(H))
        pS, cS = predict_proba_vec(_df(S))
        if not np.array_equal(cH, cS):
            alignedS = np.zeros_like(pH)
            for i, c in enumerate(cH):
                j = idx_of(cS, c)
                alignedS[i] = pS[j] if j is not None else 0.0
            pS, classes = alignedS, cH
        else:
            classes = cH

        disease_label = best_disease_label(classes, pH, pS)
        separation = float(np.max(pS - pH))
        results.append((mode, disease_label, classes, separation))

    best = max(results, key=lambda x: x[3])
    print("[CALIBRATION] THAL mode:", best[0], "disease_label:", best[1], "classes:", best[2], "sep:", best[3])
    return best[0], best[1], best[2]

THAL_MODE, DISEASE_LABEL, TRAIN_CLASSES = calibrate_thal_and_label()

def disease_probability(Xdf):
    proba, classes = predict_proba_vec(Xdf)
    if not np.array_equal(classes, TRAIN_CLASSES):
        aligned = np.zeros_like(proba)
        for i, c in enumerate(TRAIN_CLASSES):
            j = idx_of(classes, c)
            aligned[i] = proba[j] if j is not None else 0.0
        proba, classes = aligned, TRAIN_CLASSES
    d_idx = idx_of(classes, DISEASE_LABEL)
    if d_idx is None:
        d_idx = int(np.argmax(proba))
    return float(proba[d_idx])

# ------------------------------
# YES/NO Advice only
# ------------------------------
def advice_from_prob(p):
    if p >= 0.5:
        return ("❗ YES — The patient is likely to have Heart Disease.", [
            "Consult a cardiologist promptly.",
            "Get ECG, echocardiogram, and lipid profile.",
            "Avoid strenuous activity until cleared.",
            "Follow a heart-healthy diet; take prescribed meds."
        ])
    else:
        return ("✅ NO — The patient is unlikely to have Heart Disease.", [
            "Maintain a balanced diet.",
            "Exercise 30 minutes, 5 days/week.",
            "Avoid smoking; limit alcohol.",
            "Do routine health checkups."
        ])

# ------------------------------
# Build features from user form
# ------------------------------
def build_features_from_easy_form(form):
    age       = _float(form.get("age"), 45)
    sex       = _float(form.get("sex"), 1)
    cp        = _float(form.get("cp"), 0)
    exang     = _float(form.get("exang"), 0)
    bp_cat    = _float(form.get("bp_cat"), 120)
    chol_cat  = _float(form.get("chol_cat"), 200)
    hr_cat    = _float(form.get("hr_cat"), 150)
    oldpeak_c = _float(form.get("oldpeak_cat"), 0.0)
    slope     = _float(form.get("slope"), 1)

    fbs       = _float(form.get("fbs", 0), 0)
    restecg   = _float(form.get("restecg", 0), 0)
    ca        = _float(form.get("ca", 0), 0)
    thal_ui   = _float(form.get("thal", 1), 1)
    thal      = map_thal(thal_ui, THAL_MODE)

    d = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": bp_cat,
        "chol": chol_cat,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": hr_cat,
        "exang": exang,
        "oldpeak": oldpeak_c,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    return _clamp(d)

# ------------------------------
# Routes
# ------------------------------



@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        feats = build_features_from_easy_form(request.form)
        X = _df(feats)
        p = disease_probability(X)
        risk_percent = round(100.0 * p, 2)
        message, tips = advice_from_prob(p)

        print("[PREDICT DEBUG] input:", feats, "THAL_MODE:", THAL_MODE, "risk%:", risk_percent)

        return render_template(
            "result.html",
            prediction_text=message,
            risk_percent=risk_percent,
            advice=tips
        )
    except Exception as e:
        return render_template(
            "result.html",
            prediction_text=f"Error: {str(e)}",
            risk_percent=None,
            advice=[]
        )

if __name__ == "__main__":
    app.run(debug=True)
