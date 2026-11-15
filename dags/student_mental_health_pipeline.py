import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import dag, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dag(
    dag_id="student_mental_health_pipeline",
    description="Nettoie, prépare, entraîne et évalue le modèle de risque étudiant.",
    schedule="0 7 * * 1",
    start_date=pendulum.datetime(2025, 1, 1, tz="Africa/Casablanca"),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["students", "mental-health", "ml"],
)
def student_mental_health_pipeline():
    @task
    def clean_data():
        from src.pipeline import prepare_data
        path = prepare_data()
        print(f"Nettoyage terminé : {path}")
        return path

    @task
    def preprocess_data(clean_path):
        from src.pipeline import preprocess_data as split_data
        paths = split_data(clean_path)
        print(f"Prétraitement terminé : {paths}")
        return paths

    @task
    def train_model(paths):
        from src.pipeline import train_model as fit_model
        result = fit_model(paths)
        print(f"Entraînement terminé : {result}")
        return result

    @task
    def evaluate_model(paths, trained_model):
        from src.pipeline import evaluate_model as score_model
        result = score_model(paths, trained_model)
        print(f"Évaluation terminée : {result}")
        return result

    cleaned = clean_data()
    split = preprocess_data(cleaned)
    trained = train_model(split)
    evaluate_model(split, trained)


student_mental_health_pipeline()
