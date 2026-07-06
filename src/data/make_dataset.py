# src/data/make_dataset.py

from pathlib import Path
import yaml
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def load_params():
    root = Path(__file__).resolve().parents[2]

    with open(root / "params.yaml") as f:
        params = yaml.safe_load(f)

    return params["make_dataset"], root


def load_data(data_path):
    return pd.read_csv(data_path)


def split_data(df, test_size, random_state):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
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
    output_path = root / "data" / "processed"

    data = load_data(data_path)

    train, test = split_data(
        data,
        params["test_split"],
        params["seed"]
    )

    save_data(
        train,
        test,
        output_path
    )


if __name__ == "__main__":
    main()