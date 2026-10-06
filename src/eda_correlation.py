import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_PATH = "data/processed/final_dataset.csv"
OUTPUT_DIR = "reports/figures"


def main():

    print("=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(DATASET_PATH)

    # ---------------------------------------------------------
    # 1. Select numerical variables
    # ---------------------------------------------------------

    numerical_columns = [
        "Depthm",
        "T_degC",
        "Salnty",
        "O2ml_L",
        "Year",
        "Month",
        "Lat_Dec",
        "Lon_Dec",
        "Bottom_D"
    ]

    correlation_matrix = df[numerical_columns].corr()

    # ---------------------------------------------------------
    # 2. Print correlation matrix
    # ---------------------------------------------------------

    print("\nCorrelation matrix:")
    print(correlation_matrix.round(3))

    # ---------------------------------------------------------
    # 3. Correlation with target
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CORRELATION WITH T_degC")
    print("=" * 60)

    target_correlation = (
        correlation_matrix["T_degC"]
        .sort_values(ascending=False)
    )

    print(target_correlation.round(3))

    # ---------------------------------------------------------
    # 4. Heatmap
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/correlation_matrix.png",
        dpi=300
    )

    plt.close()

    print("\nCorrelation heatmap saved to:")
    print(f"{OUTPUT_DIR}/correlation_matrix.png")


if __name__ == "__main__":
    main()