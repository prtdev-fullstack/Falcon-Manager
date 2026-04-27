import streamlit as st
import json
import os
import pandas as pd

from model import train_model, predict
from utils import rule_engine, behavioral_score, combine_scores, decision_engine


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Fraud Detection Pro", layout="wide")

st.title("💳 Fraud Detection System")
st.subheader("Inspired by Falcon Fraud Manager")

HISTORY_FILE = "history.json"


# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    return train_model()

model, explainer, features = load_model()


# =========================
# HISTORY
# =========================
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history(h):
    with open(HISTORY_FILE, "w") as f:
        json.dump(h, f, indent=4)


history = load_history()


# =========================
# INPUTS
# =========================
st.sidebar.header("📥 Transaction")

amount = st.sidebar.number_input("Montant", min_value=0, step=1000)
is_international = st.sidebar.selectbox("International", [0,1])
hour = st.sidebar.slider("Heure", 0, 23, 12)
device_new = st.sidebar.selectbox("Nouveau appareil", [0,1])


data = {
    "amount": amount,
    "is_international": is_international,
    "hour": hour,
    "device_new": device_new
}


# =========================
# ANALYSE
# =========================
if st.sidebar.button("🔍 Analyser"):

    # ML
    ml_score, _ = predict(model, explainer, features, data)

    # RULES
    rule_score = rule_engine(data)

    # BEHAVIOR
    behavior = behavioral_score(history, data)

    # FINAL
    final_score = combine_scores(ml_score, rule_score, behavior)

    decision = decision_engine(final_score)

    # SAVE HISTORY
    if data not in history:
        history.append(data)
        save_history(history)


    # =========================
    # RESULTATS
    # =========================
    st.header("📊 Résultats")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Score ML", f"{ml_score:.2f}")
    col2.metric("Règles", rule_score)
    col3.metric("Comportement", behavior)
    col4.metric("Score final", f"{final_score:.2f}")

    if "BLOCK" in decision:
        st.error(decision)
    elif "REVIEW" in decision:
        st.warning(decision)
    else:
        st.success(decision)


    # =========================
    # 🧠 EXPLICATION INTELLIGENTE
    # =========================
    st.markdown("---")
    st.subheader("🧠 Explication de la décision")

    reasons = []

    # logique métier explicable
    if data["amount"] > 300000:
        reasons.append("💰 Montant élevé → risque de fraude")

    if data["is_international"] == 1:
        reasons.append("🌍 Transaction internationale → risque augmenté")

    if data["hour"] < 5 or data["hour"] > 22:
        reasons.append("🕒 Heure inhabituelle (nuit)")

    if data["device_new"] == 1:
        reasons.append("📱 Nouvel appareil non reconnu")

    if ml_score > 80:
        reasons.append("🤖 Modèle ML détecte un comportement suspect")

    if behavior > 20:
        reasons.append("👤 Comportement utilisateur anormal")

    if len(reasons) == 0:
        reasons.append("✔️ Aucun signal de fraude détecté")

    for r in reasons:
        st.write("- " + r)


# =========================
# 📊 DASHBOARD PRO FINTECH
# =========================
st.markdown("---")
st.header("📊 Dashboard Pro Fintech")

if len(history) > 0:

    df = pd.DataFrame(history)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Volume des transactions")
        st.line_chart(df["amount"])

    with col2:
        st.subheader("🌍 Transactions internationales")
        st.bar_chart(df["is_international"])

    st.subheader("📊 Analyse globale")

    st.write("📌 Total transactions :", len(df))
    st.write("💰 Montant moyen :", round(df["amount"].mean(),2))
    st.write("⚠️ Transactions suspectes :", df[df["amount"] > 300000].shape[0])

else:
    st.info("Aucune donnée dans l'historique.")