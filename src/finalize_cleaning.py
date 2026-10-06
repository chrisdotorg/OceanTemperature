import pandas as pd


INPUT_PATH = "data/processed/clean_dataset.csv"
OUTPUT_PATH = "data/processed/final_dataset.csv"


def main():

    print("=" * 60)
    print("FINALIZE CLEAN DATASET")
    print("=" * 60)

    df = pd.read_csv(INPUT_PATH)

    print("\nInitial shape:")
    print(df.shape)

    # ---------------------------------------------------------
    # 1. Detect exact duplicates
    # ---------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print("\nExact duplicate rows:")
    print(duplicate_count)

    # ---------------------------------------------------------
    # 2. Remove exact duplicates
    # ---------------------------------------------------------

    df = df.drop_duplicates().copy()

    print("\nShape after duplicate removal:")
    print(df.shape)

    # ---------------------------------------------------------
    # 3. Check target
    # ---------------------------------------------------------

    print("\nMissing target values:")
    print(df["T_degC"].isna().sum())

    # ---------------------------------------------------------
    # 4. Check remaining missing values
    # ---------------------------------------------------------

    print("\nRemaining missing values:")
    print(df.isnull().sum())

    # ---------------------------------------------------------
    # 5. Save final dataset
    # ---------------------------------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFinal dataset saved to:")
    print(OUTPUT_PATH)

    print("\nFinal shape:")
    print(df.shape)


if __name__ == "__main__":
    main()