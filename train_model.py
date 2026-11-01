"""Train and evaluate multiple TF-IDF + ML classifiers."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def load_dataset():
    fake_path = DATA_DIR / "Fake.csv"
    true_path = DATA_DIR / "True.csv"

    missing = [p.name for p in (fake_path, true_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset file(s): " + ", ".join(missing) +
            ". Download the Kaggle Fake and Real News Dataset and place both CSVs "
            "inside the dataset/ folder. See README.md."
        )

    fake = pd.read_csv(fake_path)
    real = pd.read_csv(true_path)

    for frame, label in ((fake, 0), (real, 1)):
        if "text" not in frame.columns:
            raise ValueError("Each CSV must contain a 'text' column.")

    fake["label"] = 0
    real["label"] = 1
    data = pd.concat([fake, real], ignore_index=True)

    # Combine headline and body when a title column is available.
    if "title" in data.columns:
        data["content"] = data["title"].fillna("").astype(str) + " " + data["text"].fillna("").astype(str)
    else:
        data["content"] = data["text"].fillna("").astype(str)

    data["content"] = data["content"].map(clean_text)
    data = data[data["content"].str.len() >= 30].copy()
    data = data.drop_duplicates(subset=["content"]).reset_index(drop=True)

    if data["label"].nunique() != 2:
        raise ValueError("Dataset must contain both fake and real examples.")
    return data


def build_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("classifier", classifier),
    ])


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading and preparing dataset...")
    data = load_dataset()
    X = data["content"]
    y = data["label"]

    print(f"Usable rows after cleaning and deduplication: {len(data):,}")
    print("Label counts (0 = FAKE, 1 = REAL):")
    print(y.value_counts().sort_index().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Calibrated LinearSVC provides probability estimates for the demo UI.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=4.0),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Calibrated Linear SVM": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0), cv=3
        ),
    }

    rows = []
    fitted_models = {}
    for name, classifier in models.items():
        print(f"\nTraining {name}...")
        pipeline = build_pipeline(classifier)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        result = {
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        }
        rows.append(result)
        fitted_models[name] = pipeline
        print(result)

    # Select using F1 score, not accuracy alone.
    best_row = max(rows, key=lambda row: row["f1_score"])
    best_name = best_row["model"]
    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(X_test)

    joblib.dump(best_model, MODEL_DIR / "fake_news_pipeline.joblib")

    metrics = {
        "best_model": best_name,
        "best_model_metrics": best_row,
        "comparison": rows,
        "class_mapping": {"0": "FAKE", "1": "REAL"},
        "dataset_rows_after_cleaning": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": 42,
        "test_size": 0.20,
        "classification_report": classification_report(
            y_test, best_predictions, target_names=["FAKE", "REAL"], output_dict=True
        ),
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Save confusion matrix image.
    cm = confusion_matrix(y_test, best_predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(cm, display_labels=["FAKE", "REAL"])
    display.plot(values_format="d")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    # Save model comparison chart.
    comparison = pd.DataFrame(rows).set_index("model")
    ax = comparison[["accuracy", "precision", "recall", "f1_score"]].plot(
        kind="bar", ylim=(0, 1), figsize=(10, 6)
    )
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc="lower right")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=180)
    plt.close()

    print("\nTraining complete.")
    print(f"Best model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'fake_news_pipeline.joblib'}")
    print(f"Saved metrics: {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
