import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(name: str):
    if name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=42)),
        ])
    if name == "svc_rbf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", gamma="scale", random_state=42)),
        ])
    raise ValueError(f"Unsupported model: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="svc_rbf", choices=["logistic_regression", "svc_rbf"])
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

    print(f"Confusion analysis model: {args.model}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cm).to_csv(out_dir / "confusion_matrix.csv", index=False)
        (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        print(f"Saved analysis artifacts to {out_dir}")


if __name__ == "__main__":
    main()
