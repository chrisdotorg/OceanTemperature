import pandas as pd


DATASET_PATH = "data/processed/regression_dataset.csv"


def main():

    print("=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(DATASET_PATH)

    print("\nDataset shape:")
    print(df.shape)

    # ---------------------------------------------------------
    # 1. Missing values
    # ---------------------------------------------------------

    print("\nMissing values:")
    print(df.isnull().sum())

    # ---------------------------------------------------------
    # 2. Missing percentage
    # ---------------------------------------------------------

    print("\nMissing percentage:")

    missing_percentage = (
        df.isnull().mean() * 100
    ).sort_values(ascending=False)

    print(missing_percentage)

    # ---------------------------------------------------------
    # 3. Complete observations
    # ---------------------------------------------------------

    complete_rows = df.dropna().shape[0]

    print("\nComplete observations:")
    print(complete_rows)

    print(
        "Complete percentage:",
        f"{complete_rows / len(df) * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # 4. Rows with at least one missing value
    # ---------------------------------------------------------

    incomplete_rows = df.isnull().any(axis=1).sum()

    print("\nIncomplete observations:")
    print(incomplete_rows)

    print(
        "Incomplete percentage:",
        f"{incomplete_rows / len(df) * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # 5. Missing values per row
    # ---------------------------------------------------------

    missing_per_row = df.isnull().sum(axis=1)

    print("\nMissing values per observation:")
    print(missing_per_row.value_counts().sort_index())

    # ---------------------------------------------------------
    # 6. Target missingness
    # ---------------------------------------------------------

    target_missing = df["T_degC"].isnull().sum()

    print("\nMissing target observations:")
    print(target_missing)

    # ---------------------------------------------------------
    # 7. Feature missingness excluding target
    # ---------------------------------------------------------

    features = [
        "Depthm",
        "Salnty",
        "O2ml_L",
        "Year",
        "Month",
        "Lat_Dec",
        "Lon_Dec",
        "Bottom_D"
    ]

    feature_missing = df[features].isnull().any(axis=1).sum()

    print("\nObservations with at least one missing feature:")
    print(feature_missing)

    print(
        "Percentage:",
        f"{feature_missing / len(df) * 100:.2f}%"
    )


if __name__ == "__main__":
    main()