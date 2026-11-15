import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.preprocessing import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "student_mental_health.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def prepare_data():
    df = pd.read_csv(RAW_DATA).drop_duplicates()
    df = df.dropna(subset=["Mental_Health_Risk"]).drop(columns=["Student_ID"])

    limits = {
        "Age": (16, 100), "Year_of_Study": (1, 5), "Sleep_Hours": (0, 24),
        "Study_Hours_Per_Week": (0, 44), "GPA": (0, 4),
        "Social_Media_Hours": (0, 24), "Screen_Time_Hours": (0, 24),
        "Financial_Stress": (0, 10), "Family_Support": (0, 10),
        "Attendance_Rate": (0, 100),
    }
    for col, (low, high) in limits.items():
        df.loc[(df[col] < low) | (df[col] > high), col] = np.nan

    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / "clean_students.csv"
    df.to_csv(path, index=False)
    logging.info("Données nettoyées : %s lignes", len(df))
    return str(path)


def preprocess_data(clean_path):
    df = pd.read_csv(clean_path)
    X = df.drop(columns=["Mental_Health_Risk"])
    y = df["Mental_Health_Risk"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    paths = {}
    for name, values in {
        "x_train": x_train, "x_test": x_test,
        "y_train": y_train, "y_test": y_test,
    }.items():
        path = PROCESSED / f"{name}.csv"
        values.to_csv(path, index=False, header=True)
        paths[name] = str(path)

    logging.info("Partage stratifié effectué : train=%s test=%s", x_train.shape, x_test.shape)
    return paths


def train_model(paths):
    x_train = pd.read_csv(paths["x_train"])
    y_train = pd.read_csv(paths["y_train"]).squeeze("columns")
    preprocessor = build_preprocessor(x_train)
    high_recall = make_scorer(
        recall_score, labels=["High"], average="macro", zero_division=0
    )
    candidates = {
        "SVM": (SVC(probability=True, random_state=42), {"model__C": [0.1, 1, 10]}),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]},
        ),
        "Régression Logistique": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"model__C": [0.1, 1, 10]},
        ),
    }

    best_name, best_search = None, None
    for name, (classifier, params) in candidates.items():
        pipeline = Pipeline([("preparation", preprocessor), ("model", classifier)])
        search = GridSearchCV(
            pipeline, params, scoring=high_recall, cv=3, n_jobs=1, error_score="raise"
        )
        search.fit(x_train, y_train)
        logging.info("%s : rappel High CV=%.3f, paramètres=%s", name, search.best_score_, search.best_params_)
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name, best_search = name, search

    MODELS.mkdir(parents=True, exist_ok=True)
    model_path = MODELS / "best_optimized_pipeline.joblib"
    joblib.dump(best_search.best_estimator_, model_path)
    logging.info("Modèle retenu : %s; pipeline sauvegardé dans %s", best_name, model_path)
    return {"model_path": str(model_path), "model_name": best_name}


def evaluate_model(paths, trained_model):
    x_test = pd.read_csv(paths["x_test"])
    y_test = pd.read_csv(paths["y_test"]).squeeze("columns")
    model = joblib.load(trained_model["model_path"])
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)
    labels = model.classes_

    scores = {
        "model": trained_model["model_name"],
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "recall_high": recall_score(y_test, pred, labels=["High"], average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "roc_auc_ovr": roc_auc_score(
            y_test, proba, labels=labels, multi_class="ovr", average="macro"
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")

    matrix = confusion_matrix(y_test, pred, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(REPORTS / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(xlabel="Prédit", ylabel="Réel", title=f"Matrice de confusion - {trained_model['model_name']}")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png")
    plt.close(fig)
    logging.info("Métriques de test : %s", scores)
    return str(REPORTS / "metrics.json")
