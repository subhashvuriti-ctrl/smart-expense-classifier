import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from .schema import COLUMNS
from .utils import basic_clean_description


DATA_PATH = os.path.join("data", "sample_expenses.csv")
MODEL_PATH = os.path.join("models", "expense_classifier.joblib")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[COLUMNS["description"], COLUMNS["category"]])
    df["clean_text"] = df[COLUMNS["description"]].apply(basic_clean_description)
    return df


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)),
        ("clf", LinearSVC())
    ])


def main():
    os.makedirs("models", exist_ok=True)
    df = load_data()
    X = df["clean_text"]
    y = df[COLUMNS["category"]]

    # --- dynamic test split ---
    class_counts = y.value_counts()
    n_classes = class_counts.shape[0]
    n_samples = len(y)

    test_size = 0.25
    if int(n_samples * test_size) < n_classes:
        test_size = 0.5  # ensure test set can contain all classes

    use_stratify = (class_counts.min() >= 2) and (int(n_samples * test_size) >= n_classes)
    stratify_y = y if use_stratify else None

    print("Class counts:\n", class_counts.to_string())
    print(f"Samples: {n_samples}, Classes: {n_classes}, test_size: {test_size}, stratify: {use_stratify}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    if len(y_test) > 0:
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, preds))
    else:
        print("Trained on all data (no test split due to imbalance).")

    joblib.dump(pipe, MODEL_PATH)
    print(f"✅ Model saved to → {MODEL_PATH}")


if __name__ == "__main__":
    main()
