import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Supports older notebooks that saved IQRClipper as __main__.IQRClipper.
from src.preprocessing import IQRClipper
import __main__
if not hasattr(__main__, "IQRClipper"):
    __main__.IQRClipper = IQRClipper

MODEL_PATH = ROOT / "models" / "best_optimized_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
MATRIX_PATH = ROOT / "reports" / "confusion_matrix.csv"
MATRIX_IMAGE = ROOT / "reports" / "confusion_matrix.png"

st.set_page_config(page_title="Risque psychologique étudiant", page_icon="🎓", layout="wide")
st.title("Indicateurs de risque psychologique étudiant")
st.caption("Outil éducatif d’aide à l’orientation vers les ressources de soutien. Ce résultat n’est pas un diagnostic ni une décision automatique.")


@st.cache_resource
def load_model(path, modified):
    return joblib.load(path)


if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    st.subheader("Performances du modèle")
    st.write("Modèle évalué :", metrics.get("model", "Modèle optimisé"))
    cards = st.columns(4)
    for col, key, label in zip(
        cards,
        ["accuracy", "precision_macro", "recall_high", "f1_macro"],
        ["Accuracy", "Précision macro", "Rappel High", "F1 macro"],
    ):
        col.metric(label, f"{metrics[key]:.3f}")
    chart_data = pd.DataFrame(
        {"Score": {k: v for k, v in metrics.items() if isinstance(v, (int, float))}}
    )
    st.bar_chart(chart_data, alt="Métriques du modèle sur le jeu de test")
else:
    st.info("Les métriques apparaîtront après une exécution réussie du DAG Airflow.")

left, right = st.columns(2)
with left:
    if MATRIX_PATH.exists():
        st.subheader("Matrice de confusion")
        st.dataframe(pd.read_csv(MATRIX_PATH, index_col=0), width="stretch")
with right:
    if MATRIX_IMAGE.exists():
        st.image(str(MATRIX_IMAGE), caption="Matrice de confusion sur le jeu de test", alt="Matrice de confusion du modèle")

st.subheader("Tester de nouvelles données")
upload = st.file_uploader("Importer un fichier CSV d’étudiants", type="csv")
if upload is not None:
    if not MODEL_PATH.exists():
        st.error("Aucun pipeline entraîné. Lancez d’abord le DAG Airflow.")
    else:
        try:
            model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
            new_data = pd.read_csv(upload)
            expected = list(model.feature_names_in_)
            new_data = new_data.drop(columns=["Student_ID", "Mental_Health_Risk"], errors="ignore")
            missing = [col for col in expected if col not in new_data.columns]
            if missing:
                st.error("Colonnes manquantes : " + ", ".join(missing))
            else:
                new_data = new_data[expected]
                prediction = model.predict(new_data)
                probability = model.predict_proba(new_data)
                high_index = list(model.classes_).index("High")
                result = pd.DataFrame({
                    "Risque prédit": prediction,
                    "Probabilité High": probability[:, high_index].round(3),
                })
                st.dataframe(result, width="stretch")
                st.download_button("Télécharger les prédictions", result.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as error:
            st.error(f"Impossible de traiter ce fichier : {error}")
