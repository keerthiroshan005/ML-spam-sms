from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.data_utils import load_data

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = load_data()
X = df["message"]
y = (df["label"] == "spam").astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "baseline_nb": Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 1))),
        ("clf", MultinomialNB(alpha=0.5)),
    ]),
    "improved_logreg": Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=60000)),
        ])),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)),
    ]),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print("\n", name)
    print(classification_report(y_test, pred, target_names=["ham", "spam"]))

best_name = max(results, key=lambda k: results[k]["f1"])
best_model = models[best_name]
joblib.dump(best_model, MODEL_DIR / "spam_classifier.joblib")
with (MODEL_DIR / "metrics.json").open("w") as f:
    json.dump({"best_model": best_name, "results": results}, f, indent=2)
print(f"\nSaved best model: {best_name}")
print(json.dumps(results, indent=2))
