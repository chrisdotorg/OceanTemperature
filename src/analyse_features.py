import pandas as pd


BOTTLE_PATH = "data/raw/bottle.csv"
CAST_PATH = "data/raw/cast.csv"


def main():
    bottle_features = [
        "Cst_Cnt",
        "T_degC",
        "Depthm",
        "Salnty",
        "O2ml_L",
        "O2Sat"
    ]

    cast_features = [
        "Cst_Cnt",
        "Year",
        "Month",
        "Lat_Dec",
        "Lon_Dec",
        "Distance",
        "Bottom_D"
    ]

    print("=" * 60)
    print("BOTTLE CANDIDATE FEATURES")
    print("=" * 60)

    bottle = pd.read_csv(
        BOTTLE_PATH,
        usecols=bottle_features
    )

    print("\nShape:", bottle.shape)
    print("\nMissing values:")
    print(bottle.isnull().sum())

    print("\nData types:")
    print(bottle.dtypes)

    print("\n" + "=" * 60)
    print("CAST CANDIDATE FEATURES")
    print("=" * 60)

    cast = pd.read_csv(
        CAST_PATH,
        usecols=cast_features
    )

    print("\nShape:", cast.shape)
    print("\nMissing values:")
    print(cast.isnull().sum())

    print("\nData types:")
    print(cast.dtypes)

    print("\n" + "=" * 60)
    print("JOIN KEY ANALYSIS")
    print("=" * 60)

    print("\nUnique Cst_Cnt in bottle:",
          bottle["Cst_Cnt"].nunique())

    print("Unique Cst_Cnt in cast:",
          cast["Cst_Cnt"].nunique())

    print("\nBottle rows:", len(bottle))
    print("Cast rows:", len(cast))

    print("\nDuplicate Cst_Cnt values in cast:")
    print(cast["Cst_Cnt"].duplicated().sum())

    print("\nDuplicate Cst_Cnt values in bottle:")
    print(bottle["Cst_Cnt"].duplicated().sum())
    print("\n" + "=" * 60)
    print("JOIN INTEGRITY")
    print("=" * 60)

    merged = bottle.merge(
        cast,
        on="Cst_Cnt",
        how="left",
        indicator=True
    )

    print("\nMerged shape:", merged.shape)

    print("\nJoin status:")
    print(merged["_merge"].value_counts())

    unmatched = (merged["_merge"] == "left_only").sum()

    print("\nUnmatched bottle observations:", unmatched)

    print(
        "Unmatched percentage:",
        f"{unmatched / len(bottle) * 100:.4f}%"
    )


if __name__ == "__main__":
    main()