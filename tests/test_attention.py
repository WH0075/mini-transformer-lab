import pytest
import torch

from src.minitransformer.attention import scaled_dot_product_attention


def test_scaled_dot_product_attention_output_shape() -> None:
    batch_size = 2
    seq_len = 4
    hidden_dim = 8

    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)

    output, attn_weights = scaled_dot_product_attention(q, k, v)

    assert output.shape == torch.Size([batch_size, seq_len, hidden_dim])
    assert attn_weights.shape == torch.Size([batch_size, seq_len, seq_len])


def test_scaled_dot_product_attention_weights_sum_to_one() -> None:
    batch_size = 2
    seq_len = 4
    hidden_dim = 8

    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)

    _, attn_weights = scaled_dot_product_attention(q, k, v)

    row_sums = attn_weights.sum(dim=-1)
    expected = torch.ones_like(row_sums)

    assert torch.allclose(row_sums, expected, atol=1e-6)


def test_scaled_dot_product_attention_with_mask() -> None:
    batch_size = 1
    seq_len = 3
    hidden_dim = 4

    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)

    mask = torch.tensor(
        [
            [
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 0],
            ]
        ]
    )

    _, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)

    mask_weights = attn_weights[:, :, 2]

    assert torch.allclose(mask_weights, torch.zeros_like(mask_weights), atol=1e-6)
    


def test_scaled_dot_product_attention_rejects_invalid_qk_dim() -> None:
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 7)
    v = torch.randn(2, 4, 8)

    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v)