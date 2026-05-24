import torch
import torch.nn as nn

from src.minitransformer.mlp import MLP


def creat_toy_data(num_samples: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(num_samples, 2)
    y = (x[:, 0] + x[:, 1] > 0).long()
    return x, y


def compute_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    acc = (preds == y).float().mean()
    return acc.item()


def main() -> None:
    torch.manual_seed(42)

    x, y = creat_toy_data(num_samples=1000)

    model = MLP(input_dim=2, hidden_dim=16, output_dim=2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 100

    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")
    print(model)

    model.train()

    for epoch in range(num_epochs):
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            acc = compute_accuracy(logits, y)
            print(
                f"Epoch [{epoch + 1:03d}/{num_epochs}], "
                f"loss: {loss.item():.6f}, "
                f"accuracy: {acc:.4f}"
            )
    
    final_logits = model(x)
    final_acc = compute_accuracy(final_logits, y)

    print("\nTraining finished.")
    print(f"Final accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()