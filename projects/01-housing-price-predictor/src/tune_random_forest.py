import json
from math import sqrt

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split


def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


def main():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)
    test_rmse = rmse(y_test, predictions)

    print("Best params:")
    print(search.best_params_)
    print(f"Best CV RMSE: {-search.best_score_:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    result = {
        "best_params": search.best_params_,
        "best_cv_rmse": -search.best_score_,
        "test_rmse": test_rmse,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
