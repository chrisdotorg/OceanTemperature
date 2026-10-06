import pandas as pd


DATASET_PATH = "data/processed/clean_dataset.csv"
BOTTLE_PATH = "data/raw/bottle.csv"


def main():

    print("=" * 60)
    print("DUPLICATE AND BASIC ANOMALY CHECK")
    print("=" * 60)

    df = pd.read_csv(DATASET_PATH)

    print("\nDataset shape:")
    print(df.shape)

    # ---------------------------------------------------------
    # 1. Exact duplicate rows
    # ---------------------------------------------------------

    duplicate_rows = df.duplicated().sum()

    print("\nExact duplicate rows:")
    print(duplicate_rows)

    print(
        "Duplicate percentage:",
        f"{duplicate_rows / len(df) * 100:.4f}%"
    )

    # ---------------------------------------------------------
    # 2. Target range
    # ---------------------------------------------------------

    print("\nTemperature range:")
    print("Minimum:", df["T_degC"].min())
    print("Maximum:", df["T_degC"].max())

    # ---------------------------------------------------------
    # 3. Depth range
    # ---------------------------------------------------------

    print("\nDepth range:")
    print("Minimum:", df["Depthm"].min())
    print("Maximum:", df["Depthm"].max())

    # ---------------------------------------------------------
    # 4. Salinity range
    # ---------------------------------------------------------

    print("\nSalinity range:")
    print("Minimum:", df["Salnty"].min())
    print("Maximum:", df["Salnty"].max())

    # ---------------------------------------------------------
    # 5. Oxygen range
    # ---------------------------------------------------------

    print("\nDissolved oxygen range:")
    print("Minimum:", df["O2ml_L"].min())
    print("Maximum:", df["O2ml_L"].max())

    # ---------------------------------------------------------
    # 6. Geographic range
    # ---------------------------------------------------------

    print("\nLatitude range:")
    print("Minimum:", df["Lat_Dec"].min())
    print("Maximum:", df["Lat_Dec"].max())

    print("\nLongitude range:")
    print("Minimum:", df["Lon_Dec"].min())
    print("Maximum:", df["Lon_Dec"].max())

    # ---------------------------------------------------------
    # 7. Month range
    # ---------------------------------------------------------

    print("\nMonth range:")
    print("Minimum:", df["Month"].min())
    print("Maximum:", df["Month"].max())

    # ---------------------------------------------------------
    # 8. Year range
    # ---------------------------------------------------------

    print("\nYear range:")
    print("Minimum:", df["Year"].min())
    print("Maximum:", df["Year"].max())

    # ---------------------------------------------------------
    # 9. Duplicate multiplicity
    # ---------------------------------------------------------

    duplicate_counts = (
        df[df.duplicated(keep=False)]
        .value_counts()
        .sort_values(ascending=False)
    )

    print("\nDuplicate group sizes:")
    print(duplicate_counts.value_counts().sort_index())

    # ---------------------------------------------------------
    # 10. Negative dissolved oxygen
    # ---------------------------------------------------------

    negative_oxygen = df[
        df["O2ml_L"] < 0
    ]

    print("\nNegative dissolved oxygen observations:")
    print(len(negative_oxygen))

    print("\nNegative oxygen values:")
    print(negative_oxygen["O2ml_L"].value_counts().sort_index())

    # ---------------------------------------------------------
    # 11. Very low dissolved oxygen
    # ---------------------------------------------------------

    print("\nDissolved oxygen below 0.1 mL/L:")
    print(
        (df["O2ml_L"] < 0.1).sum()
    )

    # ---------------------------------------------------------
    # 12. Extreme temperature values
    # ---------------------------------------------------------

    print("\nTemperature below 2 C:")
    print(
        (df["T_degC"] < 2).sum()
    )

    print("\nTemperature above 25 C:")
    print(
        (df["T_degC"] > 25).sum()
    )

    # ---------------------------------------------------------
    # 13. Extreme salinity values
    # ---------------------------------------------------------

    print("\nSalinity below 30:")
    print(
        (df["Salnty"] < 30).sum()
    )

    print("\nSalinity above 36:")
    print(
        (df["Salnty"] > 36).sum()
    )
    # ---------------------------------------------------------
    # 14. Inspect negative oxygen observations
    # ---------------------------------------------------------

    raw_bottle = pd.read_csv(
        BOTTLE_PATH,
        usecols=[
            "Cst_Cnt",
            "Depthm",
            "T_degC",
            "Salnty",
            "O2ml_L",
            "O_qual"
        ]
    )

    negative_oxygen_raw = raw_bottle[
        raw_bottle["O2ml_L"] < 0
    ]

    print("\n" + "=" * 60)
    print("NEGATIVE OXYGEN INVESTIGATION")
    print("=" * 60)

    print("\nNegative oxygen observations:")
    print(negative_oxygen_raw)

    print("\nOxygen quality codes:")
    print(
        negative_oxygen_raw["O_qual"]
        .value_counts(dropna=False)
    )


if __name__ == "__main__":
    main()