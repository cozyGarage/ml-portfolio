import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(name: str):
    if name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=42)),
        ])
    if name == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ])
    if name == "svc_rbf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", gamma="scale", random_state=42)),
        ])
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    raise ValueError(f"Unsupported model: {name}")


def top_confusions_from_matrix(cm: np.ndarray):
    rows = []
    for true_label in range(cm.shape[0]):
        for pred_label in range(cm.shape[1]):
            if true_label == pred_label:
                continue
            count = int(cm[true_label, pred_label])
            if count > 0:
                rows.append({"true": true_label, "pred": pred_label, "count": count})
    rows.sort(key=lambda x: x["count"], reverse=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="svc_rbf",
        choices=["logistic_regression", "knn", "svc_rbf", "random_forest"],
    )
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(args.model)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)
    top_confusions = top_confusions_from_matrix(cm)
    misclassified = [
        {"index": idx, "true": int(true), "pred": int(pred)}
        for idx, (true, pred) in enumerate(zip(y_test, preds))
        if true != pred
    ]

    print(f"Confusion analysis model: {args.model}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)
    if top_confusions:
        print("\nTop confusion pairs:")
        for row in top_confusions[:5]:
            print(f"  {row['true']} -> {row['pred']}: {row['count']}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cm).to_csv(out_dir / "confusion_matrix.csv", index=False)
        (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        pd.DataFrame(top_confusions).to_csv(out_dir / "top_confusions.csv", index=False)
        pd.DataFrame(misclassified).to_csv(out_dir / "misclassified_examples.csv", index=False)
        summary = {
            "model": args.model,
            "num_misclassified": len(misclassified),
            "top_confusions": top_confusions[:10],
        }
        (out_dir / "confusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved analysis artifacts to {out_dir}")


if __name__ == "__main__":
    main()
