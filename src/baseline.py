import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "data/processed/train_scaled.csv"
TEST_PATH = "data/processed/test_scaled.csv"

RESULTS_PATH = "reports/baseline_results.csv"

TARGET_COLUMN = "T_degC"


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    errors = y_true - y_pred

    mae = np.mean(np.abs(errors))

    mse = np.mean(errors ** 2)

    rmse = np.sqrt(mse)

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    r2 = 1 - (ss_res / ss_tot)

    return mae, mse, rmse, r2


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BASELINE MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print()
    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training observations: {len(train_df)}")
    print(f"Test observations:     {len(test_df)}")

    # --------------------------------------------------------
    # Extract target
    # --------------------------------------------------------

    y_train = train_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    # --------------------------------------------------------
    # Calculate baseline value
    # --------------------------------------------------------

    baseline_value = np.mean(y_train)

    print()
    print(f"Training mean temperature: {baseline_value:.4f} °C")

    # --------------------------------------------------------
    # Generate baseline predictions
    # --------------------------------------------------------

    y_pred = np.full(
        shape=len(y_test),
        fill_value=baseline_value
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    mae, mse, rmse, r2 = calculate_metrics(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)

    print()
    print(f"MAE  : {mae:.3f} °C")
    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse:.3f} °C")
    print(f"R²   : {r2:.4f}")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = pd.DataFrame({
        "Model": ["Mean Baseline"],
        "MAE": [mae],
        "MSE": [mse],
        "RMSE": [rmse],
        "R2": [r2]
    })

    results.to_csv(
        RESULTS_PATH,
        index=False
    )

    print()
    print("Results saved to:")
    print(RESULTS_PATH)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()