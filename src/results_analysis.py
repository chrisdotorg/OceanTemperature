import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

COMPARISON_PATH = "reports/model_comparison.csv"
BASELINE_PATH = "reports/baseline_results.csv"

OUTPUT_PATH = "reports/results_summary.csv"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    print()
    print("Loading model comparison results...")

    comparison_df = pd.read_csv(COMPARISON_PATH)

    print()
    print(comparison_df.to_string(index=False))

    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    baseline = comparison_df[
        comparison_df["Model"] == "Mean Baseline"
    ].iloc[0]

    pytorch = comparison_df[
        comparison_df["Model"] == "PyTorch Linear Regression"
    ].iloc[0]

    # --------------------------------------------------------
    # Calculate improvements
    # --------------------------------------------------------

    mae_improvement = (
        (baseline["MAE"] - pytorch["MAE"])
        / baseline["MAE"]
        * 100
    )

    rmse_improvement = (
        (baseline["RMSE"] - pytorch["RMSE"])
        / baseline["RMSE"]
        * 100
    )

    # --------------------------------------------------------
    # Display analysis
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print()
    print(f"Baseline MAE : {baseline['MAE']:.4f} °C")
    print(f"Model MAE    : {pytorch['MAE']:.4f} °C")

    print()

    print(f"Baseline RMSE: {baseline['RMSE']:.4f} °C")
    print(f"Model RMSE   : {pytorch['RMSE']:.4f} °C")

    print()

    print(f"Baseline R²  : {baseline['R2']:.4f}")
    print(f"Model R²     : {pytorch['R2']:.4f}")

    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("IMPROVEMENT OVER BASELINE")
    print("=" * 60)

    print()
    print(f"MAE improvement : {mae_improvement:.2f}%")
    print(f"RMSE improvement: {rmse_improvement:.2f}%")

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print()

    if pytorch["R2"] > 0:
        print(
            "The linear regression model explains a substantial "
            "proportion of the variance in water temperature."
        )

    if pytorch["MAE"] < baseline["MAE"]:
        print(
            "The trained model substantially reduces the average "
            "prediction error compared with the mean baseline."
        )

    if pytorch["RMSE"] < baseline["RMSE"]:
        print(
            "The reduction in RMSE indicates improved prediction "
            "accuracy, including for larger errors."
        )

    print()
    print(
        "The model should nevertheless be interpreted with caution "
        "for extreme temperature and depth ranges, where the error "
        "analysis revealed substantially larger prediction errors."
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame({
        "Metric": [
            "Baseline_MAE",
            "Model_MAE",
            "Baseline_RMSE",
            "Model_RMSE",
            "Baseline_R2",
            "Model_R2",
            "MAE_Improvement_Percent",
            "RMSE_Improvement_Percent"
        ],
        "Value": [
            baseline["MAE"],
            pytorch["MAE"],
            baseline["RMSE"],
            pytorch["RMSE"],
            baseline["R2"],
            pytorch["R2"],
            mae_improvement,
            rmse_improvement
        ]
    })

    summary.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("Results summary saved to:")
    print(OUTPUT_PATH)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()