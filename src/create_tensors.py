import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


TRAIN_PATH = "data/processed/train_scaled.csv"
VALIDATION_PATH = "data/processed/validation_scaled.csv"
TEST_PATH = "data/processed/test_scaled.csv"

BATCH_SIZE = 1024

FEATURE_COLUMNS = [
    "Depthm",
    "Salnty",
    "O2ml_L",
    "Year",
    "Lat_Dec",
    "Lon_Dec",
    "Month_sin",
    "Month_cos"
]

TARGET = "T_degC"


def load_dataset(path):

    df = pd.read_csv(path)

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET].values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    return X, y


def main():

    print("=" * 60)
    print("PYTORCH DATA PREPARATION")
    print("=" * 60)

    print("\nLoading datasets...")

    X_train, y_train = load_dataset(TRAIN_PATH)
    X_validation, y_validation = load_dataset(VALIDATION_PATH)
    X_test, y_test = load_dataset(TEST_PATH)

    print("\nTensor shapes:")

    print(f"X_train:       {X_train.shape}")
    print(f"y_train:       {y_train.shape}")

    print(f"X_validation:  {X_validation.shape}")
    print(f"y_validation:  {y_validation.shape}")

    print(f"X_test:        {X_test.shape}")
    print(f"y_test:        {y_test.shape}")

    print("\nTensor types:")

    print(f"X_train: {X_train.dtype}")
    print(f"y_train: {y_train.dtype}")

    print("\nCreating TensorDatasets...")

    train_dataset = TensorDataset(X_train, y_train)
    validation_dataset = TensorDataset(X_validation, y_validation)
    test_dataset = TensorDataset(X_test, y_test)

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")

    print("\nCreating DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"Batch size: {BATCH_SIZE}")

    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Test batches:       {len(test_loader)}")

    # Inspect one training batch
    X_batch, y_batch = next(iter(train_loader))

    print("\nExample training batch:")

    print(f"X batch shape: {X_batch.shape}")
    print(f"y batch shape: {y_batch.shape}")

    print("\nFirst observation:")
    print(f"Features: {X_batch[0]}")
    print(f"Target:   {y_batch[0].item()}")

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nAvailable device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()