import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASELINE_PATH = "reports/baseline_results.csv"

EVALUATION_PATH = "reports/evaluation_results.csv"

COMPARISON_PATH = "reports/model_comparison.csv"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    print()
    print("Loading model results...")

    baseline = pd.read_csv(
        BASELINE_PATH
    )

    pytorch = pd.read_csv(
        EVALUATION_PATH
    )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    comparison = pd.concat(
        [baseline, pytorch],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Display comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print()

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Extract metrics
    # --------------------------------------------------------

    baseline_mae = comparison.loc[
        comparison["Model"] == "Mean Baseline",
        "MAE"
    ].iloc[0]

    model_mae = comparison.loc[
        comparison["Model"] == "PyTorch Linear Regression",
        "MAE"
    ].iloc[0]

    baseline_rmse = comparison.loc[
        comparison["Model"] == "Mean Baseline",
        "RMSE"
    ].iloc[0]

    model_rmse = comparison.loc[
        comparison["Model"] == "PyTorch Linear Regression",
        "RMSE"
    ].iloc[0]

    # --------------------------------------------------------
    # Calculate improvements
    # --------------------------------------------------------

    mae_improvement = (
        (baseline_mae - model_mae)
        / baseline_mae
        * 100
    )

    rmse_improvement = (
        (baseline_rmse - model_rmse)
        / baseline_rmse
        * 100
    )

    # --------------------------------------------------------
    # Display improvements
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IMPROVEMENT OVER BASELINE")
    print("=" * 70)

    print()
    print(
        f"MAE improvement  : "
        f"{mae_improvement:.2f}%"
    )

    print(
        f"RMSE improvement : "
        f"{rmse_improvement:.2f}%"
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    comparison.to_csv(
        COMPARISON_PATH,
        index=False
    )

    print()
    print("Comparison saved to:")
    print(COMPARISON_PATH)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()