import math
import torch



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


def main() -> None:
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    hidden_dim = 8

    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)    

    mask = create_causal_mask(seq_len)
    output, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)

    print(f"q.shape: {q.shape}")
    print(f"k.shape: {k.shape}")
    print(f"v.shape: {v.shape}")
    print(f"mask.shape: {mask.shape}")
    print("mask:")
    print(mask)
    print(f"attn_weights.shape: {attn_weights.shape}")
    print(f"output.shape: {output.shape}")

    row_sums = attn_weights.sum(dim=-1)
    print(f"attn_weights row sums: {row_sums}")

    future_weights = torch.triu(attn_weights, diagonal=1)
    print("future attention weights:")
    print(future_weights)

    assert mask.shape == (1, seq_len, seq_len)
    assert output.shape == q.shape
    assert attn_weights.shape == (batch_size, seq_len, seq_len)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    assert torch.allclose(
        future_weights,
        torch.zeros_like(future_weights),
        atol=1e-6,
    )

    print("\nCausal attention check passed!")



if __name__ == "__main__":
    main()