import pandas as pd


BOTTLE_PATH = "data/raw/bottle.csv"
CAST_PATH = "data/raw/cast.csv"
OUTPUT_PATH = "data/processed/regression_dataset.csv"


def main():

    print("=" * 60)
    print("BUILDING REGRESSION DATASET")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load selected Bottle features
    # ---------------------------------------------------------

    bottle_columns = [
        "Cst_Cnt",
        "T_degC",
        "Depthm",
        "Salnty",
        "O2ml_L"
    ]

    print("\nLoading bottle data...")

    bottle = pd.read_csv(
        BOTTLE_PATH,
        usecols=bottle_columns
    )

    print("Bottle shape:", bottle.shape)

    # ---------------------------------------------------------
    # 2. Load selected Cast features
    # ---------------------------------------------------------

    cast_columns = [
        "Cst_Cnt",
        "Year",
        "Month",
        "Lat_Dec",
        "Lon_Dec",
        "Bottom_D"
    ]

    print("\nLoading cast data...")

    cast = pd.read_csv(
        CAST_PATH,
        usecols=cast_columns
    )

    print("Cast shape:", cast.shape)

    # ---------------------------------------------------------
    # 3. Merge datasets
    # ---------------------------------------------------------

    print("\nMerging datasets...")

    dataset = bottle.merge(
        cast,
        on="Cst_Cnt",
        how="left"
    )

    print("Merged shape:", dataset.shape)

    # ---------------------------------------------------------
    # 4. Remove technical join key
    # ---------------------------------------------------------

    dataset = dataset.drop(columns=["Cst_Cnt"])

    # ---------------------------------------------------------
    # 5. Display resulting structure
    # ---------------------------------------------------------

    print("\nFinal columns:")
    for column in dataset.columns:
        print(f"  - {column}")

    print("\nMissing values:")
    print(dataset.isnull().sum())

    # ---------------------------------------------------------
    # 6. Save working dataset
    # ---------------------------------------------------------

    dataset.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nDataset saved to:")
    print(OUTPUT_PATH)

    print("\nFinal dataset shape:", dataset.shape)


if __name__ == "__main__":
    main()