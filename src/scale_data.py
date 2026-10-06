import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


TRAIN_PATH = "data/processed/train_imputed.csv"
VALIDATION_PATH = "data/processed/validation_imputed.csv"
TEST_PATH = "data/processed/test_imputed.csv"

TRAIN_OUTPUT = "data/processed/train_scaled.csv"
VALIDATION_OUTPUT = "data/processed/validation_scaled.csv"
TEST_OUTPUT = "data/processed/test_scaled.csv"

SCALER_OUTPUT = "data/processed/feature_scaler.pkl"

TARGET = "T_degC"


def main():

    print("=" * 60)
    print("FEATURE STANDARDIZATION")
    print("=" * 60)

    print("\nLoading imputed datasets...")

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    test = pd.read_csv(TEST_PATH)

    print(f"Training shape: {train.shape}")
    print(f"Validation shape: {validation.shape}")
    print(f"Test shape: {test.shape}")

    # Separate features and target
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

    X_train = train[feature_columns]
    X_validation = validation[feature_columns]
    X_test = test[feature_columns]

    y_train = train[TARGET]
    y_validation = validation[TARGET]
    y_test = test[TARGET]

    print("\nFeatures:")
    for feature in feature_columns:
        print(f" - {feature}")

    print(f"\nTarget:")
    print(f" - {TARGET}")

    # Create scaler
    scaler = StandardScaler()

    print("\nFitting StandardScaler on training data...")

    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform validation and test using training statistics
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames
    train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=feature_columns
    )

    validation_scaled = pd.DataFrame(
        X_validation_scaled,
        columns=feature_columns
    )

    test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=feature_columns
    )

    # Add target back without scaling
    train_scaled[TARGET] = y_train.values
    validation_scaled[TARGET] = y_validation.values
    test_scaled[TARGET] = y_test.values

    # Save scaler
    with open(SCALER_OUTPUT, "wb") as file:
        pickle.dump(scaler, file)

    # Save datasets
    train_scaled.to_csv(TRAIN_OUTPUT, index=False)
    validation_scaled.to_csv(VALIDATION_OUTPUT, index=False)
    test_scaled.to_csv(TEST_OUTPUT, index=False)

    print("\nTraining feature statistics after scaling:")

    print(
        train_scaled[feature_columns]
        .agg(["mean", "std"])
        .round(4)
    )

    print("\nDatasets saved:")

    print(f" - {TRAIN_OUTPUT}")
    print(f" - {VALIDATION_OUTPUT}")
    print(f" - {TEST_OUTPUT}")
    print(f" - {SCALER_OUTPUT}")

    print("\nFinal shapes:")
    print(f"Training: {train_scaled.shape}")
    print(f"Validation: {validation_scaled.shape}")
    print(f"Test: {test_scaled.shape}")


if __name__ == "__main__":
    main()