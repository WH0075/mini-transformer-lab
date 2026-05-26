from pathlib import Path
import logging
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

from src.minitransformer.mlp import MLP
from src.minitransformer.datasets import ToyClassificationDataset


def load_config():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "mlp_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("mlp_train")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


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


def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
) -> tuple[float, float]:
    
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def save_checkpoint(
        checkpoint_path: Path,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        best_val_acc: float,
        config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config,
        },
        checkpoint_path,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    config = load_config()
    torch.manual_seed(config["seed"])

    log_dir = project_root / "logs"
    checkpoint_dir = project_root / "checkpoints"

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "mlp_train.log"
    checkpoint_path = checkpoint_dir / "mlp_best.pt"

    logger = setup_logger(log_path)

    logger.info("Load config:")
    for key, value in config.items():
        logger.info(f" {key}: {value}")

    dataset = ToyClassificationDataset(num_samples=config["num_samples"])

    train_size = int(len(dataset) * config["train_ratio"])
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
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

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Validation size: {len(val_dataset)}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info("Model:")
    logger.info(model)

    best_val_acc = 0.0
    num_epochs = config["num_epochs"]

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
        )

        logger.info(
            f"Epoch [{epoch + 1:03d}/{num_epochs}]"
            f"train_loss: {train_loss:.6f}"
            f"train_acc: {train_acc:.4f}"
            f"val_loss: {val_loss:.6f}"
            f"val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            save_checkpoint(
                checkpoint_path=checkpoint_path,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                best_val_acc=best_val_acc,
                config=config,
            )

            logger.info(
                f"Saved best checkpoint to {checkpoint_path}"
                f"with val_acc: {best_val_acc:.4f}"
            )
        
    logger.info("Training finished.")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")

        


if __name__ == "__main__":
    main()