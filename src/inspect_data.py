import pandas as pd

CAST_PATH = "data/raw/cast.csv"
BOTTLE_PATH = "data/raw/bottle.csv"


def inspect_dataset(path, name):
    print("\n" + "=" * 60)
    print(f"{name} DATASET")
    print("=" * 60)

    df = pd.read_csv(path, nrows=1000)

    print(f"\nShape of sample: {df.shape}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    for column, count in missing.items():
        percentage = (count / len(df)) * 100
        print(f"  - {column}: {count} ({percentage:.2f}%)")

    print("\nFirst 5 rows:")
    print(df.head())


def main():
    inspect_dataset(CAST_PATH, "CAST")
    inspect_dataset(BOTTLE_PATH, "BOTTLE")

    print("\n" + "=" * 60)
    print("GLOBAL BOTTLE DATASET DIAGNOSTICS")
    print("=" * 60)

    columns = [
        "T_degC",
        "Salnty",
        "Depthm",
        "O2ml_L",
        "O2Sat",
        "T_qual",
        "S_qual",
        "O_qual",
    ]

    bottle = pd.read_csv(
        BOTTLE_PATH,
        usecols=columns
    )

    print(f"\nFull dataset shape: {bottle.shape}")

    print("\nMissing values:")
    missing = bottle.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    for column, count in missing.items():
        percentage = (count / len(bottle)) * 100
        print(f"  - {column}: {count} ({percentage:.2f}%)")

    print("\nTarget candidate statistics:")
    print(
        bottle[
            ["T_degC", "Salnty", "Depthm", "O2ml_L"]
        ].describe()
    )

    print("\nTemperature quality codes:")
    print(
        bottle["T_qual"]
        .value_counts(dropna=False)
        .sort_index()
    )


if __name__ == "__main__":
    main()