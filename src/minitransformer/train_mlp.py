from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.minitransformer.mlp import MLP
from src.minitransformer.datasets import ToyClassificationDataset


def load_config() -> int:
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "mlp_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy

def main() -> None:
    config = load_config()

    torch.manual_seed(config["seed"])

    dataset = ToyClassificationDataset(num_samples=config["num_samples"])

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    model = MLP(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
    )


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    print("Loaded config:")
    for key, value in config.items():
        print(f" {key}: {value}")

    print("\nDataset and DataLoader:")
    print(f" dataset size: {len(dataset)}")

    first_x, first_y = dataset[0]
    print(f" single x shape: {first_x.shape}")
    print(f" single y shape: {first_y.shape}")

    first_batch_x, first_batch_y = next(iter(dataloader))
    print(f" batch x shape: {first_batch_x.shape}")
    print(f" batch y shape: {first_batch_y.shape}")

    print("\nModel:")
    print(model)

    num_epochs = config["num_epochs"]

    for epoch in range(num_epochs):
        avg_loss, accuracy = train_one_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1:03d}/{num_epochs}], "
                f"loss: {avg_loss:.6f}, "
                f"accuracy: {accuracy:.4f}"
            )
    
    print("\nTraining finished.")
    print(f"Final accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()