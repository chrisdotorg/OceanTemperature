import pandas as pd
from sklearn.impute import SimpleImputer


INPUT_PATH = "data/processed/features_dataset.csv"

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/validation.csv"
TEST_PATH = "data/processed/test.csv"


TARGET = "T_degC"


def main():
    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    print("\nLoading dataset...")
    df = pd.read_csv(INPUT_PATH)

    print(f"Dataset shape: {df.shape}")

    # ---------------------------------------------------------
    # 2. Chronological split
    # ---------------------------------------------------------

    df = df.sort_values("Year").reset_index(drop=True)

    years = sorted(df["Year"].unique())

    train_end = int(len(years) * 0.70)
    val_end = int(len(years) * 0.85)

    train_years = years[:train_end]
    val_years = years[train_end:val_end]
    test_years = years[val_end:]

    train_df = df[df["Year"].isin(train_years)].copy()
    val_df = df[df["Year"].isin(val_years)].copy()
    test_df = df[df["Year"].isin(test_years)].copy()

    print("\nSplit results:")

    print(
        f"Training:   {train_years[0]}-{train_years[-1]} "
        f"({len(train_df)} observations)"
    )

    print(
        f"Validation: {val_years[0]}-{val_years[-1]} "
        f"({len(val_df)} observations)"
    )

    print(
        f"Test:       {test_years[0]}-{test_years[-1]} "
        f"({len(test_df)} observations)"
    )

    # ---------------------------------------------------------
    # 3. Separate features and target
    # ---------------------------------------------------------

    feature_columns = [
        "Depthm",
        "Salnty",
        "O2ml_L",
        "Year",
        "Lat_Dec",
        "Lon_Dec",
        "Month_sin",
        "Month_cos"
    ]

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET]

    X_val = val_df[feature_columns]
    y_val = val_df[TARGET]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET]

    # ---------------------------------------------------------
    # 4. Missing-value analysis
    # ---------------------------------------------------------

    print("\nMissing values before imputation:")

    print("\nTraining:")
    print(X_train.isnull().sum())

    print("\nValidation:")
    print(X_val.isnull().sum())

    print("\nTest:")
    print(X_test.isnull().sum())

    # ---------------------------------------------------------
    # 5. Fit imputer ONLY on training data
    # ---------------------------------------------------------

    print("\nFitting median imputer on training data...")

    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train)

    X_val_imputed = imputer.transform(X_val)

    X_test_imputed = imputer.transform(X_test)

    # ---------------------------------------------------------
    # 6. Display learned medians
    # ---------------------------------------------------------

    print("\nLearned training medians:")

    for feature, median in zip(feature_columns, imputer.statistics_):
        print(f"  {feature}: {median}")

    # ---------------------------------------------------------
    # 7. Verify missing values are gone
    # ---------------------------------------------------------

    print("\nMissing values after imputation:")

    print(
        f"Training:   {pd.isna(X_train_imputed).sum()}"
    )

    print(
        f"Validation: {pd.isna(X_val_imputed).sum()}"
    )

    print(
        f"Test:       {pd.isna(X_test_imputed).sum()}"
    )

    # ---------------------------------------------------------
    # 8. Convert back to DataFrames
    # ---------------------------------------------------------

    X_train_imputed = pd.DataFrame(
        X_train_imputed,
        columns=feature_columns,
        index=train_df.index
    )

    X_val_imputed = pd.DataFrame(
        X_val_imputed,
        columns=feature_columns,
        index=val_df.index
    )

    X_test_imputed = pd.DataFrame(
        X_test_imputed,
        columns=feature_columns,
        index=test_df.index
    )

    # Add target back
    X_train_imputed[TARGET] = y_train.values
    X_val_imputed[TARGET] = y_val.values
    X_test_imputed[TARGET] = y_test.values

    # ---------------------------------------------------------
    # 9. Save imputed datasets
    # ---------------------------------------------------------

    X_train_imputed.to_csv(
        "data/processed/train_imputed.csv",
        index=False
    )

    X_val_imputed.to_csv(
        "data/processed/validation_imputed.csv",
        index=False
    )

    X_test_imputed.to_csv(
        "data/processed/test_imputed.csv",
        index=False
    )

    print("\nImputed datasets saved:")
    print("  - data/processed/train_imputed.csv")
    print("  - data/processed/validation_imputed.csv")
    print("  - data/processed/test_imputed.csv")


if __name__ == "__main__":
    main()