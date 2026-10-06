import pandas as pd


DATASET_PATH = "data/processed/final_dataset.csv"


def main():

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(DATASET_PATH)

    print("\nDataset shape:")
    print(df.shape)

    # ---------------------------------------------------------
    # 1. Data types
    # ---------------------------------------------------------

    print("\nData types:")
    print(df.dtypes)

    # ---------------------------------------------------------
    # 2. Descriptive statistics
    # ---------------------------------------------------------

    print("\nDescriptive statistics:")
    print(df.describe().T)

    # ---------------------------------------------------------
    # 3. Target statistics
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TARGET VARIABLE: T_degC")
    print("=" * 60)

    print(df["T_degC"].describe())

    # ---------------------------------------------------------
    # 4. Quantiles
    # ---------------------------------------------------------

    print("\nTemperature quantiles:")

    quantiles = df["T_degC"].quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )

    print(quantiles)

    # ---------------------------------------------------------
    # 5. Unique values
    # ---------------------------------------------------------

    print("\nUnique values:")

    for column in df.columns:
        print(
            f"{column:10s}: {df[column].nunique()}"
        )


if __name__ == "__main__":
    main()