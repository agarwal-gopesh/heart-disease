from pathlib import Path
import yaml
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from dvclive import Live


def load_params():

    root = Path(__file__).resolve().parents[2]

    with open(root / "params.yaml", "r") as f:
        params = yaml.safe_load(f)

    return params["make_dataset"], root


def load_data(data_path):

    data = pd.read_csv(data_path)

    return data


def split_data(df, test_split, seed):

    train, test = train_test_split(
        df,
        test_size=test_split,
        random_state=seed
    )

    return train, test


def save_data(train, test, output_path):

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        output_path / "train.csv",
        index=False
    )

    test.to_csv(
        output_path / "test.csv",
        index=False
    )


def main():

    params, root = load_params()

    input_file = sys.argv[1]

    data_path = root / input_file

    data = load_data(
        data_path
    )

    with Live() as live:

        # experiment parameters
        live.log_param(
            "test_split",
            params["test_split"]
        )

        live.log_param(
            "seed",
            params["seed"]
        )

        train, test = split_data(
            data,
            params["test_split"],
            params["seed"]
        )

    save_data(
        train,
        test,
        root / "data" / "processed"
    )


if __name__ == "__main__":
    main()