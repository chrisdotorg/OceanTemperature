import pandas as pd


DATASET_PATH = "data/processed/regression_dataset.csv"


def main():

    print("=" * 60)
    print("MISSING VALUE PATTERN ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(DATASET_PATH)

    columns = [
        "T_degC",
        "Salnty",
        "O2ml_L",
        "Bottom_D"
    ]

    # ---------------------------------------------------------
    # 1. Missingness correlation
    # ---------------------------------------------------------

    missing = df[columns].isnull()

    print("\nMissingness correlation:")
    print(missing.corr())

    # ---------------------------------------------------------
    # 2. Pairwise missing values
    # ---------------------------------------------------------

    print("\nPairwise missing counts:")

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            col1 = columns[i]
            col2 = columns[j]

            count = (
                missing[col1] &
                missing[col2]
            ).sum()

            print(f"{col1} + {col2}: {count}")

    # ---------------------------------------------------------
    # 3. Complete target observations
    # ---------------------------------------------------------

    valid_target = df["T_degC"].notna()

    print("\nObservations with valid target:")
    print(valid_target.sum())

    print(
        "Percentage:",
        f"{valid_target.mean() * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # 4. Missing features among valid targets
    # ---------------------------------------------------------

    feature_columns = [
        "Salnty",
        "O2ml_L",
        "Bottom_D"
    ]

    valid_target_df = df.loc[
        valid_target,
        feature_columns
    ]

    print("\nFeature missingness after removing")
    print("observations with missing target:")

    print(valid_target_df.isnull().sum())

    print(
        "\nComplete feature rows:",
        valid_target_df.dropna().shape[0]
    )

    print(
        "Complete feature percentage:",
        f"{valid_target_df.dropna().shape[0] / len(valid_target_df) * 100:.2f}%"
    )


if __name__ == "__main__":
    main()