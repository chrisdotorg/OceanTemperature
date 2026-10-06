import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model import LinearRegressionModel


# ============================================================
# CONFIGURATION
# ============================================================

TEST_PATH = "data/processed/test_scaled.csv"
MODEL_PATH = "data/processed/best_model.pth"

PREDICTIONS_PATH = "reports/test_predictions.csv"
EVALUATION_RESULTS_PATH = "reports/evaluation_results.csv"

ACTUAL_PREDICTED_FIGURE = (
    "reports/figures/actual_vs_predicted.png"
)

RESIDUALS_FIGURE = (
    "reports/figures/residuals.png"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
    print("MODEL EVALUATION")
    print("=" * 60)

    print()
    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print()
    print("Loading test dataset...")

    test_df = pd.read_csv(TEST_PATH)

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

    target_column = "T_degC"

    X_test = test_df[feature_columns].values
    y_test = test_df[target_column].values

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(DEVICE)

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.float32
    ).view(-1, 1).to(DEVICE)

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print("Loading trained model...")

    model = LinearRegressionModel(
        input_features=len(feature_columns)
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print()
    print("Generating predictions...")

    with torch.no_grad():

        predictions = model(
            X_test_tensor
        )

    y_pred = (
        predictions
        .cpu()
        .numpy()
        .flatten()
    )

    y_true = (
        y_test_tensor
        .cpu()
        .numpy()
        .flatten()
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    mae, mse, rmse, r2 = calculate_metrics(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print()
    print(f"MAE  : {mae:.3f} °C")
    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse:.3f} °C")
    print(f"R²   : {r2:.4f}")

    # --------------------------------------------------------
    # Save evaluation metrics
    # --------------------------------------------------------

    evaluation_results = pd.DataFrame({
        "Model": ["PyTorch Linear Regression"],
        "MAE": [mae],
        "MSE": [mse],
        "RMSE": [rmse],
        "R2": [r2]
    })

    evaluation_results.to_csv(
        EVALUATION_RESULTS_PATH,
        index=False
    )

    print()
    print("Evaluation results saved to:")
    print(EVALUATION_RESULTS_PATH)

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_df = pd.DataFrame({
        "Actual_Temperature": y_true,
        "Predicted_Temperature": y_pred,
        "Residual": y_true - y_pred
    })

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    print()
    print("Predictions saved to:")
    print(PREDICTIONS_PATH)

    # ========================================================
    # VISUALIZATION
    # ========================================================

    # --------------------------------------------------------
    # Random sample for visualization
    # --------------------------------------------------------

    sample_size = min(
        5000,
        len(y_true)
    )

    rng = np.random.default_rng(42)

    sample_indices = rng.choice(
        len(y_true),
        size=sample_size,
        replace=False
    )

    y_actual_sample = y_true[sample_indices]
    y_pred_sample = y_pred[sample_indices]

    # --------------------------------------------------------
    # Figure 1: Actual vs Predicted
    # --------------------------------------------------------

    plt.figure(figsize=(10, 7))

    plt.scatter(
        y_actual_sample,
        y_pred_sample,
        alpha=0.25,
        s=12
    )

    min_value = min(
        y_actual_sample.min(),
        y_pred_sample.min()
    )

    max_value = max(
        y_actual_sample.max(),
        y_pred_sample.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=2
    )

    plt.xlabel(
        "Actual Temperature (°C)"
    )

    plt.ylabel(
        "Predicted Temperature (°C)"
    )

    plt.title(
        "Actual vs Predicted Water Temperature"
    )

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        ACTUAL_PREDICTED_FIGURE,
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Figure 2: Residual Analysis
    # --------------------------------------------------------

    residuals = y_true - y_pred

    residual_sample = residuals[sample_indices]
    predicted_sample = y_pred[sample_indices]

    plt.figure(figsize=(10, 7))

    plt.scatter(
        predicted_sample,
        residual_sample,
        alpha=0.25,
        s=12
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=2
    )

    plt.xlabel(
        "Predicted Temperature (°C)"
    )

    plt.ylabel(
        "Residual (°C)"
    )

    plt.title(
        "Residual Analysis"
    )

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        RESIDUALS_FIGURE,
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Completion message
    # --------------------------------------------------------

    print()
    print("Figures saved:")

    print(ACTUAL_PREDICTED_FIGURE)
    print(RESIDUALS_FIGURE)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()