import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# === CHARGEMENT DU MODÈLE & DONNÉES ===
model = joblib.load('pipeline_complete.pkl')
data = pd.read_csv('data_test.csv')
data_train = pd.read_csv('data_train.csv')

# Nettoyage des colonnes
def clean_columns(df):
    df.columns = [re.sub(r'\W+', '_', col) for col in df.columns]
    return df

def encode_categorical_columns(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
    return df

data = encode_categorical_columns(clean_columns(data))
data_train = encode_categorical_columns(clean_columns(data_train))

data_scaled = data.copy()
data_train_scaled = data_train.copy()

explainer = shap.TreeExplainer(model['model'])

# === INTERFACE UTILISATEUR ===
st.sidebar.header("Sélection du client")
client_id = st.sidebar.selectbox("Choisissez un client:", data['SK_ID_CURR'])

st.title("Dashboard Crédit Client")
st.write("Analyse détaillée du risque de crédit.")

if client_id in data['SK_ID_CURR'].values:
    client_data = data[data['SK_ID_CURR'] == client_id]
    st.subheader("Informations du client")
    st.write(client_data)

    info_client = client_data.drop('TARGET', axis=1)
    prediction = model.predict_proba(info_client)[0][1]
    threshold = 0.09
    decision = "Accepté" if prediction < threshold else "Refusé"
    decision_color = "green" if decision == "Accepté" else "red"
    st.markdown(f"<h3 style='color:{decision_color};'>Décision: {decision} ({prediction:.2%})</h3>", unsafe_allow_html=True)


    # Explication SHAP Globale
    st.subheader("Explication Globale (SHAP)")
    shap_vals_global = explainer.shap_values(data_scaled.drop('SK_ID_CURR', axis=1))
    if isinstance(shap_vals_global, list):
        shap_vals_global = shap_vals_global[1]
    X_global = data_scaled.drop('SK_ID_CURR', axis=1)
    shap_values_exp = shap.Explanation(values=shap_vals_global, data=X_global, feature_names=X_global.columns)
    
    fig_global, ax_global = plt.subplots()
    shap.plots.beeswarm(shap_values_exp, max_display=20, show=False)
    st.pyplot(fig_global)

    # Explication SHAP Locale
    st.subheader("Explication Locale (SHAP)")
    X_client = client_data.drop('SK_ID_CURR', axis=1)
    shap_values_local = explainer(X_client)
    
    if isinstance(shap_values_local, list):
        shap_for_class1 = shap_values_local[1]
        local_explanation = shap_for_class1[0]
    else:
        local_explanation = shap_values_local[0]
    
    fig_local = plt.figure()
    shap.waterfall_plot(local_explanation, show=False)
    st.pyplot(fig_local)

else:
    st.error("Client introuvable.")
