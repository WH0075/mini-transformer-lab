from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.minitransformer.datasets import CharLanguageModelDataset
from src.minitransformer.model import TinyTransformerLanguageModel
from src.minitransformer.tokenizer import CharTokenizer, load_text


def build_config() -> dict[str, Any]:

    return {
        "block_size": 16,
        "batch_size": 16,
        "embed_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "feedforward_dim": 128,
        "learning_rate": 1e-3,
        "num_epochs": 50,
        "max_new_tokens": 80,
        "seed": 42,
    }


def train_one_epoch(
    model: TinyTransformerLanguageModel,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """训练模型一个 epoch。"""

    model.train()

    total_loss = 0.0
    total_token = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits, _ = model(batch_x)

        batch_size, seq_len, vocab_size = logits.shape

        loss = loss_fn(
            logits.view(batch_size * seq_len, vocab_size),
            batch_y.view(batch_size * seq_len),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_tokens = batch_size * seq_len
        total_loss += loss.item() * num_tokens
        total_token += num_tokens
    
    return total_loss / total_token


@torch.no_grad()
def generate_text(
    model: TinyTransformerLanguageModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    block_size: int,
    device: torch.device,
) -> str:
    """根据 prompt 生成文本。"""

    model.eval()
    
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        context = input_ids[:, -block_size:]

        logits, _ = model(context)

        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_id], dim=1)
    
    generated_ids = input_ids.squeeze(0).tolist()
    return tokenizer.decode(generated_ids)



def save_checkpoint(
    checkpoint_path: Path,
    model: TinyTransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    tokenizer: CharTokenizer,
    epoch: int,
    train_loss: float,
) -> None:
    """保存训练 checkpoint。"""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "vocab_size": tokenizer.vocab_size,
        },
        checkpoint_path,
    )
   


def main() -> None:
    """训练 Tiny Transformer 语言模型。"""
    config = build_config()
    
    torch.manual_seed(config["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parents[2]
    corpus_path = project_root / "data" / "raw" / "tiny_corpus.txt"
    checkpoint_path = project_root / "checkpoints" / "tiny_transformer_lm.pt"
    
    text = load_text(corpus_path)
    tokenizer = CharTokenizer(text)

    token_ids = tokenizer.encode(text)

    dataset = CharLanguageModelDataset(
        token_ids=token_ids,
        block_size=config["block_size"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    model = TinyTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        block_size=config["block_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        feedforward_dim=config["feedforward_dim"],
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
    )

    print("Training TinyTransformerLanguageModel")
    print(f"Corpus path: {corpus_path}")
    print(f"Text length: {len(text)}")
    print(f"Number of token ids: {len(token_ids)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Block size: {config['block_size']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Num epochs: {config['num_epochs']}")
    print()

    for epoch in range(1, config["num_epochs"] + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        

        if epoch == 1 or epoch % 10 == 0:
            sample = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt="h",
                max_new_tokens=config["max_new_tokens"],
                block_size=config["block_size"],
                device=device,
            )

            print(
            f"Epoch [{epoch:03d}/{config['num_epochs']}], "
            f"train_loss: {train_loss:.6f}"
            )
            print("Generated sample:")
            print(sample)
            print("-" * 60)

    final_loss = train_loss

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        epoch=config["num_epochs"],
        train_loss=final_loss,
    )

    print()
    print(f"Checkpoint saved to: {checkpoint_path}")

    final_sample = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="h",
        max_new_tokens=config["max_new_tokens"],
        block_size=config["block_size"],
        device=device,
    )

    print()
    print("Final generated sample:")
    print(final_sample)



if __name__ == "__main__":
    main()