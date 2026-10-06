import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "data/processed/train_scaled.csv"
VALIDATION_PATH = "data/processed/validation_scaled.csv"

BATCH_SIZE = 1024
LEARNING_RATE = 0.001
EPOCHS = 50

MODEL_PATH = "data/processed/best_model.pth"


# ============================================================
# MODEL
# ============================================================

class LinearRegressionModel(nn.Module):

    def __init__(self, input_features):
        super().__init__()

        self.linear = nn.Linear(
            in_features=input_features,
            out_features=1
        )

    def forward(self, x):
        return self.linear(x)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(path):

    df = pd.read_csv(path)

    X = df.drop(columns=["T_degC"]).values
    y = df["T_degC"].values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    return X, y


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 60)
    print("LINEAR REGRESSION TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading datasets...")

    X_train, y_train = load_dataset(TRAIN_PATH)
    X_validation, y_validation = load_dataset(VALIDATION_PATH)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")

    print(f"Number of features: {X_train.shape[1]}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_dataset = TensorDataset(X_train, y_train)
    validation_dataset = TensorDataset(
        X_validation,
        y_validation
    )

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LinearRegressionModel(
        input_features=X_train.shape[1]
    ).to(device)

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(f"\nLoss function: {criterion}")
    print(f"Optimizer: Adam")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_validation_loss = float("inf")

    history = []

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(EPOCHS):

        # ====================================================
        # TRAINING
        # ====================================================

        model.train()

        train_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(X_batch)

            # Calculate loss
            loss = criterion(predictions, y_batch)

            # Backpropagation
            loss.backward()

            # Update parameters
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_dataset)

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for X_batch, y_batch in validation_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch)

                loss = criterion(
                    predictions,
                    y_batch
                )

                validation_loss += (
                    loss.item() * X_batch.size(0)
                )

        validation_loss /= len(validation_dataset)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss
            }
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            saved = "*"

        else:
            saved = ""

        # ====================================================
        # PROGRESS
        # ====================================================

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Validation Loss: {validation_loss:.6f} "
            f"{saved}"
        )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        "reports/training_history.csv",
        index=False
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"\nBest validation MSE: {best_validation_loss:.6f}")

    print(f"\nBest model saved to:")
    print(MODEL_PATH)

    print("\nTraining history saved to:")
    print("reports/training_history.csv")


if __name__ == "__main__":
    main()