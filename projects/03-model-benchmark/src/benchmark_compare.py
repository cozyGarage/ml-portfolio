import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_dataset(name: str):
    if name == "wine":
        return load_wine(as_frame=True)
    if name == "breast_cancer":
        return load_breast_cancer(as_frame=True)
    if name == "iris":
        return load_iris(as_frame=True)
    raise ValueError(f"Unsupported dataset: {name}")


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
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wine", "breast_cancer", "iris"], default="wine")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    dataset = get_dataset(args.dataset)
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in build_models().items():
        acc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        test_acc = accuracy_score(y_test, preds)
        test_macro_f1 = f1_score(y_test, preds, average="macro")

        results[name] = {
            "cv_accuracy_mean": float(np.mean(acc_scores)),
            "cv_accuracy_std": float(np.std(acc_scores)),
            "cv_macro_f1_mean": float(np.mean(f1_scores)),
            "cv_macro_f1_std": float(np.std(f1_scores)),
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_macro_f1),
        }
        print(name)
        print(f"  cv_accuracy_mean: {np.mean(acc_scores):.4f}")
        print(f"  cv_macro_f1_mean: {np.mean(f1_scores):.4f}")
        print(f"  test_accuracy: {test_acc:.4f}")
        print(f"  test_macro_f1: {test_macro_f1:.4f}")

    best_model = max(results, key=lambda k: results[k]["cv_macro_f1_mean"])
    print(f"Best model by CV macro F1: {best_model}")

    payload = {
        "dataset": args.dataset,
        "best_model": best_model,
        "results": results,
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved benchmark results to {out}")


if __name__ == "__main__":
    main()
