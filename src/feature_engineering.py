import pandas as pd
import numpy as np


INPUT_PATH = "data/processed/final_dataset.csv"
OUTPUT_PATH = "data/processed/features_dataset.csv"


def main():
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Load cleaned dataset
    print("\nLoading cleaned dataset...")
    df = pd.read_csv(INPUT_PATH)

    print(f"Initial shape: {df.shape}")

    # Create cyclical month features
    print("\nCreating cyclical month features...")

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # Select final features and target
    selected_columns = [
        "Depthm",
        "Salnty",
        "O2ml_L",
        "Year",
        "Lat_Dec",
        "Lon_Dec",
        "Month_sin",
        "Month_cos",
        "T_degC"
    ]

    df = df[selected_columns]

    print("\nFinal features:")
    for column in selected_columns[:-1]:
        print(f" - {column}")

    print("\nTarget:")
    print(" - T_degC")

    print(f"\nFinal shape: {df.shape}")

    print("\nMissing values:")
    print(df.isnull().sum())

    # Save feature-engineered dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDataset saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()