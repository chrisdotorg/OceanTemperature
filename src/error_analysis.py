import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTIONS_PATH = "reports/test_predictions.csv"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    print()
    print("Loading test predictions...")

    df = pd.read_csv(PREDICTIONS_PATH)

    print(f"Number of observations: {len(df)}")

    # --------------------------------------------------------
    # Basic error statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RESIDUAL STATISTICS")
    print("=" * 60)

    residuals = df["Residual"]

    print()
    print(f"Mean residual   : {residuals.mean():.4f} °C")
    print(f"Std residual    : {residuals.std():.4f} °C")
    print(f"Minimum residual: {residuals.min():.4f} °C")
    print(f"Maximum residual: {residuals.max():.4f} °C")

    # --------------------------------------------------------
    # Absolute errors
    # --------------------------------------------------------

    df["Absolute_Error"] = np.abs(df["Residual"])

    print()
    print("=" * 60)
    print("ABSOLUTE ERROR STATISTICS")
    print("=" * 60)

    print()
    print(f"Mean absolute error   : {df['Absolute_Error'].mean():.4f} °C")
    print(f"Median absolute error : {df['Absolute_Error'].median():.4f} °C")
    print(f"Maximum absolute error: {df['Absolute_Error'].max():.4f} °C")

    # --------------------------------------------------------
    # Largest prediction errors
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LARGEST PREDICTION ERRORS")
    print("=" * 60)

    largest_errors = (
        df.sort_values(
            "Absolute_Error",
            ascending=False
        )
        .head(10)
    )

    print()

    print(
        largest_errors[
            [
                "Actual_Temperature",
                "Predicted_Temperature",
                "Residual",
                "Absolute_Error"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Error percentiles
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ERROR PERCENTILES")
    print("=" * 60)

    percentiles = [50, 75, 90, 95, 99]

    for percentile in percentiles:

        value = np.percentile(
            df["Absolute_Error"],
            percentile
        )

        print(
            f"{percentile}th percentile: "
            f"{value:.4f} °C"
        )
    # --------------------------------------------------------
    # Error analysis by temperature range
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ERROR BY ACTUAL TEMPERATURE RANGE")
    print("=" * 60)

    df["Temperature_Range"] = pd.cut(
        df["Actual_Temperature"],
        bins=[-np.inf, 5, 10, 15, 20, np.inf],
        labels=[
            "< 5 °C",
            "5–10 °C",
            "10–15 °C",
            "15–20 °C",
            ">= 20 °C"
        ]
    )

    print()

    for temperature_range, group in df.groupby(
        "Temperature_Range",
        observed=False
    ):

        errors = group["Residual"]

        mae = np.mean(
            np.abs(errors)
        )

        rmse = np.sqrt(
            np.mean(errors ** 2)
        )

        mean_residual = errors.mean()

        print(f"{temperature_range}")
        print(f"  Observations    : {len(group)}")
        print(f"  MAE             : {mae:.4f} °C")
        print(f"  RMSE            : {rmse:.4f} °C")
        print(f"  Mean residual   : {mean_residual:.4f} °C")
        print()

        # --------------------------------------------------------
    # Error analysis by depth
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ERROR BY DEPTH RANGE")
    print("=" * 60)

    # Load depth information from the test dataset
    test_features = pd.read_csv(
        "data/processed/test_scaled.csv"
    )

    # IMPORTANT:
    # test_scaled.csv contains standardized Depthm.
    # We therefore use the original test dataset to recover
    # the actual depth values.

    original_test = pd.read_csv(
        "data/processed/test_imputed.csv"
    )

    df["Depthm"] = original_test["Depthm"].values

    df["Depth_Range"] = pd.cut(
        df["Depthm"],
        bins=[-np.inf, 50, 100, 500, 1000, np.inf],
        labels=[
            "0–50 m",
            "50–100 m",
            "100–500 m",
            "500–1000 m",
            ">1000 m"
        ]
    )

    print()

    for depth_range, group in df.groupby(
        "Depth_Range",
        observed=False
    ):

        errors = group["Residual"]

        mae = np.mean(
            np.abs(errors)
        )

        rmse = np.sqrt(
            np.mean(errors ** 2)
        )

        mean_residual = errors.mean()

        print(f"{depth_range}")
        print(f"  Observations    : {len(group)}")
        print(f"  MAE             : {mae:.4f} °C")
        print(f"  RMSE            : {rmse:.4f} °C")
        print(f"  Mean residual   : {mean_residual:.4f} °C")
        print()

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()