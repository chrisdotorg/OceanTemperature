import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_PATH = "data/processed/final_dataset.csv"
OUTPUT_DIR = "reports/figures"


def plot_histogram(df, column, title, filename):

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df[column],
        bins=50,
        kde=True
    )

    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{filename}",
        dpi=300
    )

    plt.close()


def plot_boxplot(df, column, title, filename):

    plt.figure(figsize=(10, 4))

    sns.boxplot(
        x=df[column]
    )

    plt.title(title)
    plt.xlabel(column)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{filename}",
        dpi=300
    )

    plt.close()


def main():

    print("=" * 60)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(DATASET_PATH)

    # ---------------------------------------------------------
    # Histograms
    # ---------------------------------------------------------

    histogram_columns = {
        "T_degC": "Water Temperature Distribution",
        "Depthm": "Depth Distribution",
        "Salnty": "Salinity Distribution",
        "O2ml_L": "Dissolved Oxygen Distribution",
        "Year": "Observation Year Distribution",
        "Month": "Observation Month Distribution",
        "Bottom_D": "Bottom Depth Distribution"
    }

    for column, title in histogram_columns.items():

        print(f"Generating histogram: {column}")

        plot_histogram(
            df,
            column,
            title,
            f"{column.lower()}_histogram.png"
        )

    # ---------------------------------------------------------
    # Boxplots
    # ---------------------------------------------------------

    boxplot_columns = [
        "T_degC",
        "Depthm",
        "Salnty",
        "O2ml_L",
        "Bottom_D"
    ]

    for column in boxplot_columns:

        print(f"Generating boxplot: {column}")

        plot_boxplot(
            df,
            column,
            f"{column} Boxplot",
            f"{column.lower()}_boxplot.png"
        )

    print("\nAll figures saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()