import torch
import torch.nn as nn

from src.minitransformer.attention import create_causal_mask
from src.minitransformer.blocks import TransformerBlock


class TinyTransformerLanguageModel(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
    ) -> None:
        
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if block_size <= 0:
            raise ValueError("block_size must be positive.")

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive.")

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feedforward_dim = feedforward_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.final_ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:

        if input_ids.dim() != 2:
            raise ValueError("Input_ids must have shape [B, T].")

        batch_size, seq_len = input_ids.shape

        if seq_len > self.block_size:
            raise ValueError("Sequence length exceeds block_size.")
        
        token_emb = self.token_embedding(input_ids)
        
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.position_embedding(positions)
        
        x = token_emb + pos_emb
        mask = create_causal_mask(seq_len).to(input_ids.device)

        all_attn_weights: list[torch.Tensor] = []
        
        for block in self.blocks:
            x, atten_weights = block(x, mask=mask)
            all_attn_weights.append(atten_weights)

        x = self.final_ln(x)
        logits = self.lm_head(x)

        return logits, all_attn_weights


def main() -> None:
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    vocab_size = 30
    block_size = 8
    embed_dim = 32
    num_heads = 4
    num_layers = 2
    feedforward_dim = 128

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
    )

    model = TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
    )

    logits, all_attn_weights = model(input_ids)

    print(f"input_ids.shape: {input_ids.shape}")
    print(f"logits.shape: {logits.shape}")
    print(f"number of layers: {len(all_attn_weights)}")

    for i, attn_weights in enumerate(all_attn_weights):
        print(f"Layer {i} attn_weights.shape: {attn_weights.shape}")

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert len(all_attn_weights) == num_layers

    for attn_weights in all_attn_weights:
        assert attn_weights.shape == (
            batch_size,
            num_heads,
            seq_len,
            seq_len,
        )

        future_weights = torch.triu(attn_weights, diagonal=1)
        assert torch.allclose(
            future_weights,
            torch.zeros_like(future_weights),
            atol=1e-6,
        )

    print("\nTinyTransformerLanguageModel check passed!")


if __name__ == "__main__":
    main()