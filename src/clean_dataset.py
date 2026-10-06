import pandas as pd


BOTTLE_PATH = "data/raw/bottle.csv"
CAST_PATH = "data/raw/cast.csv"

OUTPUT_PATH = "data/processed/clean_dataset.csv"


def main():

    print("=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load Bottle data
    # ---------------------------------------------------------

    bottle_columns = [
        "Cst_Cnt",
        "T_degC",
        "Depthm",
        "Salnty",
        "O2ml_L",
        "T_qual"
    ]

    print("\nLoading bottle data...")

    bottle = pd.read_csv(
        BOTTLE_PATH,
        usecols=bottle_columns
    )

    print("Initial observations:", len(bottle))

    # ---------------------------------------------------------
    # 2. Inspect temperature quality codes
    # ---------------------------------------------------------

    print("\nTemperature quality codes:")
    print(bottle["T_qual"].value_counts(dropna=False))

    # ---------------------------------------------------------
    # 3. Remove missing target
    # ---------------------------------------------------------

    missing_target = bottle["T_degC"].isna().sum()

    print("\nMissing target observations:", missing_target)

    bottle = bottle[
        bottle["T_degC"].notna()
    ].copy()

    print(
        "After removing missing target:",
        len(bottle)
    )

    # ---------------------------------------------------------
    # 4. Remove questionable temperature observations
    # ---------------------------------------------------------

    questionable = (
        bottle["T_qual"] == 8
    ).sum()

    print(
        "\nQuestionable temperature observations:",
        questionable
    )

    bottle = bottle[
        bottle["T_qual"] != 8
    ].copy()

    print(
        "After removing questionable observations:",
        len(bottle)
    )

    # ---------------------------------------------------------
    # 5. Load Cast data
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

    # ---------------------------------------------------------
    # 6. Merge Bottle and Cast
    # ---------------------------------------------------------

    print("\nMerging datasets...")

    dataset = bottle.merge(
        cast,
        on="Cst_Cnt",
        how="left"
    )

    print("Merged shape:", dataset.shape)

    # ---------------------------------------------------------
    # 7. Remove technical columns
    # ---------------------------------------------------------

    dataset = dataset.drop(
        columns=[
            "Cst_Cnt",
            "T_qual"
        ]
    )

    # ---------------------------------------------------------
    # 8. Display final structure
    # ---------------------------------------------------------

    print("\nFinal columns:")

    for column in dataset.columns:
        print(f"  - {column}")

    print("\nMissing values:")
    print(dataset.isnull().sum())

    # ---------------------------------------------------------
    # 9. Save cleaned dataset
    # ---------------------------------------------------------

    dataset.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nDataset saved to:")
    print(OUTPUT_PATH)

    print("\nFinal dataset shape:")
    print(dataset.shape)


if __name__ == "__main__":
    main()