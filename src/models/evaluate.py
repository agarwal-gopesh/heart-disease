from pathlib import Path
import yaml
import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from dvclive import Live


def load_params():

    root = Path(__file__).resolve().parents[2]

    with open(root / "params.yaml", "r") as f:
        params = yaml.safe_load(f)

    return params["evaluate"], root


def load_data(root):

    test = pd.read_csv(
        root / "data" / "features" / "test_features.csv"
    )

    return test


def split_xy(df):

    X = df.drop(
        columns=["trip_duration"]
    )

    y = df["trip_duration"]

    return X, y


def load_model(root):

    model = joblib.load(
        root / "models" / "model.pkl"
    )

    return model


def evaluate(model, X_test, y_test):

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    import numpy as np

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    return mae, rmse, r2


def main():

    params, root = load_params()

    test = load_data(root)

    X_test, y_test = split_xy(
        test
    )

    model = load_model(
        root
    )

    mae, rmse, r2 = evaluate(
        model,
        X_test,
        y_test
    )

    with Live() as live:

        live.log_metric(
            "MAE",
            mae
        )

        live.log_metric(
            "RMSE",
            rmse
        )

        live.log_metric(
            "R2",
            r2
        )
        live.next_step()


if __name__ == "__main__":
    main()