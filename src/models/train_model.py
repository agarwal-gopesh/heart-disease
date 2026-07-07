from pathlib import Path
import yaml
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from dvclive import Live


def load_params():

    root = Path(__file__).resolve().parents[2]

    with open(root / "params.yaml", "r") as f:
        params = yaml.safe_load(f)

    return params["train_model"], root


def load_data(root):

    train = pd.read_csv(
        root / "data" / "features" / "train_features.csv"
    )

    return train


def split_xy(df):

    X = df.drop(
        columns=["trip_duration"]
    )

    y = df["trip_duration"]

    return X, y


def get_model(params):

    model_name = params["model_name"]

    if model_name == "random_forest":

        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=params["seed"]
        )

    elif model_name == "xgboost":

        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=params["seed"]
        )

    elif model_name == "lightgbm":

        model = LGBMRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=params["seed"]
        )

    elif model_name == "decision_tree":

        model = DecisionTreeRegressor(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=params["seed"]
        )

    else:

        raise ValueError(
            f"{model_name} not supported"
        )

    return model


def save_model(model, root):

    model_dir = root / "models"

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        model_dir / "model.pkl"
    )


def main():

    params, root = load_params()

    train = load_data(root)

    X_train, y_train = split_xy(train)

    with Live() as live:

        # log parameters

        live.log_param(
            "model_name",
            params["model_name"]
        )

        live.log_param(
            "n_estimators",
            params["n_estimators"]
        )

        live.log_param(
            "max_depth",
            params["max_depth"]
        )

        live.log_param(
            "min_samples_split",
            params["min_samples_split"]
        )

        live.log_param(
            "learning_rate",
            params["learning_rate"]
        )

        live.log_param(
            "seed",
            params["seed"]
        )

        model = get_model(params)

        model.fit(
            X_train,
            y_train
        )

    save_model(
        model,
        root
    )


if __name__ == "__main__":
    main()