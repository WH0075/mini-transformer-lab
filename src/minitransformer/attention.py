import math
import torch
import torch.nn as nn


def create_causal_mask(seq_len: int) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0)


def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    if q.size(-1) != k.size(-1):
        raise ValueError("q and k must have the same last dimension.")

    d_k = q.size(-1)

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    
    attn_weights = torch.softmax(scores, dim=-1)
    output = attn_weights @ v

    return output, attn_weights


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split [B, T, C] into [B, H, T, D]."""
        batch_size, seq_len, embed_dim = x.shape
        
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)

        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge [B, H, T, D] into [B, T, C]."""
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_len, num_heads * head_dim)
        
        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len, embed_dim = x.shape
        
        if embed_dim != self.embed_dim:
            raise ValueError("Input embed_dim does not match module embed_dim")
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        attn_output, attn_weights = scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
        )
        
        attn_output = self._merge_heads(attn_output)
        output = self.out_proj(attn_output)
        
        return output, attn_weights


def main() -> None:
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    embed_dim = 32
    num_heads = 4

    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = create_causal_mask(seq_len)   

    attention = MultiHeadSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
    )

    output, attn_weights = attention(x, mask=mask)

    print(f"x.shape: {x.shape}")
    print(f"mask.shape: {mask.shape}")
    print(f"output.shape: {output.shape}")
    print(f"attn_weights.shape: {attn_weights.shape}")
    print(f"num_heads: {attention.num_heads}")
    print(f"head_dim: {attention.head_dim}")

    future_weights = torch.triu(attn_weights, diagonal=1)

    assert output.shape == x.shape
    assert attn_weights.shape == (
        batch_size,
        num_heads,
        seq_len,
        seq_len,
    )
    assert torch.allclose(
        future_weights,
        torch.zeros_like(future_weights),
        atol=1e-6,
    )

    print("\nMulti-head self-attention check passed!")



if __name__ == "__main__":
    main()





