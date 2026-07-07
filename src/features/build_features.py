from pathlib import Path
import yaml
import pandas as pd


def load_params():

    root = Path(__file__).resolve().parents[2]

    with open(root / "params.yaml", "r") as f:
        params = yaml.safe_load(f)

    return params["build_features"], root


def load_data(train_path, test_path):

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


def encode_features(df):

    # encode store_and_fwd_flag
    df["store_and_fwd_flag"] = (
        df["store_and_fwd_flag"]
        .map({"N": 0, "Y": 1})
    )

    return df


def save_data(train, test, output_path):

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        output_path / "train_features.csv",
        index=False
    )

    test.to_csv(
        output_path / "test_features.csv",
        index=False
    )


def main():

    params, root = load_params()

    train_path = root / params["train_path"]
    test_path = root / params["test_path"]

    train, test = load_data(
        train_path,
        test_path
    )

    train = encode_features(train)
    test = encode_features(test)

    output_path = root / "data" / "features"

    save_data(
        train,
        test,
        output_path
    )


if __name__ == "__main__":
    main()