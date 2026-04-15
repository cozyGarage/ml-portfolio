from math import sqrt

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


def main():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(rmse, greater_is_better=False)

    models = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
        rmse_scores = -scores
        print(name)
        print(f"  mean_rmse: {np.mean(rmse_scores):.4f}")
        print(f"  std_rmse: {np.std(rmse_scores):.4f}")


if __name__ == "__main__":
    main()
