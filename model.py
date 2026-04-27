import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap


# =========================
# DATA
# =========================
def load_data():
    try:
        df = pd.read_csv("data.csv")
        if len(df.columns) == 1:
            df = pd.read_csv("data.csv", sep=";")
    except:
        df = generate_data()

    required = {"amount","is_international","hour","device_new","is_fraud"}

    if not required.issubset(df.columns):
        df = generate_data()

    return df.dropna()


def generate_data():
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "amount": np.random.randint(1000, 500000, n),
        "is_international": np.random.randint(0,2,n),
        "hour": np.random.randint(0,24,n),
        "device_new": np.random.randint(0,2,n),
    })

    df["is_fraud"] = (
        (df["amount"] > 300000) |
        (df["device_new"] == 1) & (df["hour"] > 22)
    ).astype(int)

    return df


# =========================
# TRAIN
# =========================
def train_model():
    df = load_data()

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)

    return model, explainer, list(X.columns)


# =========================
# SAFE SHAP
# =========================
def get_shap(explainer, df):

    shap_values = explainer.shap_values(df)

    if isinstance(shap_values, list):
        values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        values = shap_values[0]

    values = np.array(values)

    return values


# =========================
# PREDICT
# =========================
def predict(model, explainer, features, data):

    df = pd.DataFrame([data])

    proba = model.predict_proba(df)[0][1] * 100

    shap_values = get_shap(explainer, df)

    explanation = {}

    for i, f in enumerate(features):
        try:
            explanation[f] = float(shap_values[i])
        except:
            explanation[f] = 0.0

    return round(proba,2), explanation