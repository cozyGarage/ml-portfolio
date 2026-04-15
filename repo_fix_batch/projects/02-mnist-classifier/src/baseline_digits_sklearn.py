import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=42)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "svc_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", gamma="scale", random_state=42)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Classes: {np.unique(y).tolist()}")

    results = {}
    best_name = None
    best_score = -1.0

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        results[name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "classification_report": classification_report(y_test, preds),
        }
        print(f"{name}: accuracy={acc:.4f} macro_f1={macro_f1:.4f}")
        if macro_f1 > best_score:
            best_score = macro_f1
            best_name = name

    print(f"Best model: {best_name} (macro_f1={best_score:.4f})")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump({"best_model": best_name, "results": results}, f, indent=2)
        print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
