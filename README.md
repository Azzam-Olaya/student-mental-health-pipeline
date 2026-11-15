# student-mental-health-pipeline
# 🎓 Student Wellbeing Risk Classifier

A complete Data Science pipeline to analyze the factors associated with student
wellbeing and mental health risk, and to build a supervised classification prototype.

> ⚠️ **Disclaimer**: this project is strictly educational. The model is in no way
> a diagnostic tool, a clinical screening tool, a basis for disciplinary decisions,
> or a student selection mechanism.

## 🎯 Context and Objectives

An educational institution wants to use a student dataset in order to:

- **Detect** profiles with an increased risk of psychological difficulties
- **Anticipate** support needs (prevention)
- **Understand** the factors associated with psychological risk
- **Guide** students toward appropriate resources
- **Prioritize** student support actions

## 🔄 Data Science Workflow Covered

| Step | Content |
|------|---------|
| 1. EDA | Data structure, missing values, duplicates, outliers, descriptive statistics, target distribution, correlations |
| 2. Preparation | 80/20 split, imputation, outlier handling (IQR/Z-score), encoding, SMOTE, scaling, leakage-free pipeline |
| 3. Modeling | Comparison of at least 3 classifiers (accuracy, precision, recall, F1, confusion matrix, ROC-AUC) |
| 4. Optimization | GridSearchCV / RandomizedSearchCV, model selection, error analysis, pipeline export (`.joblib`) |
| 5. Deployment | Airflow DAG (scheduled orchestration) + interactive Streamlit app |

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · Imbalanced-learn · Matplotlib / Seaborn ·
Joblib · Apache Airflow · Streamlit

## 📁 Project Structure

​```
├── data/              # Raw and cleaned data
├── notebooks/         # EDA and experiments
├── src/               # Preprocessing, training, and evaluation scripts
├── dags/              # Airflow DAG
├── app/               # Streamlit application
├── models/            # Saved pipeline (.joblib)
├── reports/           # Figures and results
└── requirements.txt
​```

## 🚀 Quick Start

​```bash
pip install -r requirements.txt
streamlit run app/app.py
​```

## ⚖️ Limitations and Ethics

Predictions are statistical indicators derived from a limited dataset. They never
replace the judgment of a healthcare professional or a qualified counselor.

## Étape 5 — Orchestration et application

Le DAG `student_mental_health_pipeline` nettoie les données, crée un partage stratifié, optimise les trois modèles, évalue le meilleur sur le jeu de test et sauvegarde le pipeline complet. Il est planifié chaque lundi à 7 h (Africa/Casablanca). Airflow conserve l’état des tâches, leurs journaux et retente une tâche en échec une fois.

L’application Streamlit affiche les métriques et la matrice de confusion produites par Airflow. Elle permet aussi d’importer un fichier CSV contenant les variables explicatives et de consulter les risques prédits.

Depuis la racine du projet :

```bash
docker compose up --build airflow streamlit
```

Ouvrir Airflow sur `http://localhost:8080` et l’application sur `http://localhost:8501`. Le compte administrateur Airflow est généré au premier démarrage; consulter `docker compose logs airflow` pour trouver le mot de passe. Le pipeline applique l’imputation, l’encodage et la standardisation dans les modèles, après le partage des données, afin d’éviter les fuites entre entraînement et test. Les prédictions sont des indicateurs statistiques pour orienter une revue de soutien; elles ne constituent pas un diagnostic.
