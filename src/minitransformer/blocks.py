import torch
import torch.nn as nn

from src.minitransformer.attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    
    def __init__(self, embed_dim: int, feedforward_dim: int) -> None: 
        super().__init__()
        
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        
        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive.")
        
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, self.feedforward_dim),
            nn.GELU(),
            nn.Linear(self.feedforward_dim, self.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
    ) -> None:

        super().__init__()
        
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        
        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive.")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        
        self.ln2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            feedforward_dim=feedforward_dim,
        )
        pass

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if x.size(-1) != self.embed_dim:
            raise ValueError("Input embed_dim does not match block embed_dim.")
        
        attn_output, attn_weights = self.attention(
            self.ln1(x),
            mask=mask,
        )
        
        x = x + attn_output
        ffn_output = self.feed_forward(self.ln2(x))
        
        x = x + ffn_output
        
        return x, attn_weights


def main() -> None:

    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    embed_dim = 32
    num_heads = 4
    feedforward_dim = 128

    x = torch.randn(batch_size, seq_len, embed_dim)
    
    from src.minitransformer.attention import create_causal_mask
    
    mask = create_causal_mask(seq_len)
    
    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
    )
    
    output, attn_weights = block(x, mask=mask)

    print(f"x.shape: {x.shape}")
    print(f"mask.shape: {mask.shape}")
    print(f"output.shape: {output.shape}")
    print(f"attn_weights.shape: {attn_weights.shape}")

    assert output.shape == x.shape
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

    print("\nTransformer block check passed!")


if __name__ == "__main__":
    main()