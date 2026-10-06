import torch
import torch.nn as nn


class LinearRegressionModel(nn.Module):

    def __init__(self, input_features):
        super().__init__()

        self.linear = nn.Linear(
            input_features,
            1
        )

    def forward(self, x):

        return self.linear(x)


def main():

    print("=" * 60)
    print("LINEAR REGRESSION MODEL")
    print("=" * 60)

    input_features = 8

    model = LinearRegressionModel(input_features)

    print("\nModel architecture:")
    print(model)

    print("\nModel parameters:")

    for name, parameter in model.named_parameters():
        print(f"{name}:")
        print(parameter)

    print("\nNumber of trainable parameters:")

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(total_parameters)

    # Test forward pass
    X = torch.randn(5, input_features)

    predictions = model(X)

    print("\nTest input shape:")
    print(X.shape)

    print("\nPrediction shape:")
    print(predictions.shape)

    print("\nPredictions:")
    print(predictions)


if __name__ == "__main__":
    main()