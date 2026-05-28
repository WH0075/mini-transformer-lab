import pytest
import torch

from src.minitransformer.attention import (
    MultiHeadSelfAttention,
    create_causal_mask,
    scaled_dot_product_attention,
)


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

    masked_weights = attn_weights[:, :, 2]

    assert torch.allclose(
        masked_weights,
        torch.zeros_like(masked_weights),
        atol=1e-6,
    )


def test_scaled_dot_product_attention_rejects_invalid_qk_dim() -> None:
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 7)
    v = torch.randn(2, 4, 8)

    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v)


def test_create_causal_mask_shape() -> None:
    seq_len = 4

    mask = create_causal_mask(seq_len)

    assert mask.shape == torch.Size([1, seq_len, seq_len])


def test_create_causal_mask_values() -> None:
    mask = create_causal_mask(4)

    expected = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ]
    )

    assert torch.equal(mask, expected)


def test_create_causal_mask_rejects_invalid_seq_len() -> None:
    with pytest.raises(ValueError):
        create_causal_mask(0)


def test_causal_mask_blocks_future_attention() -> None:
    batch_size = 2
    seq_len = 4
    hidden_dim = 8

    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)

    mask = create_causal_mask(seq_len)

    _, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)

    future_weights = torch.triu(attn_weights, diagonal=1)

    assert torch.allclose(
        future_weights,
        torch.zeros_like(future_weights),
        atol=1e-6,
    )


def test_causal_attention_weights_sum_to_one() -> None:
    batch_size = 2
    seq_len = 4
    hidden_dim = 8

    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)

    mask = create_causal_mask(seq_len)

    _, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)

    row_sums = attn_weights.sum(dim=-1)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    )


def test_multi_head_self_attention_output_shape() -> None:
    batch_size = 2
    seq_len = 4
    embed_dim = 32
    num_heads = 4

    x = torch.randn(batch_size, seq_len, embed_dim)

    attention = MultiHeadSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
    )

    output, attn_weights = attention(x)

    assert output.shape == torch.Size([batch_size, seq_len, embed_dim])
    assert attn_weights.shape == torch.Size(
        [batch_size, num_heads, seq_len, seq_len]
    )


def test_multi_head_self_attention_head_dim() -> None:
    attention = MultiHeadSelfAttention(embed_dim=32, num_heads=4)

    assert attention.head_dim == 8
    assert attention.embed_dim == 32
    assert attention.num_heads == 4


def test_multi_head_self_attention_rejects_invalid_embed_dim() -> None:
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(embed_dim=30, num_heads=4)


def test_multi_head_self_attention_rejects_invalid_input_dim() -> None:
    batch_size = 2
    seq_len = 4
    wrong_embed_dim = 16

    attention = MultiHeadSelfAttention(embed_dim=32, num_heads=4)
    x = torch.randn(batch_size, seq_len, wrong_embed_dim)

    with pytest.raises(ValueError):
        attention(x)


def test_multi_head_self_attention_with_causal_mask() -> None:
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

    future_weights = torch.triu(attn_weights, diagonal=1)

    assert output.shape == torch.Size([batch_size, seq_len, embed_dim])
    assert attn_weights.shape == torch.Size(
        [batch_size, num_heads, seq_len, seq_len]
    )
    assert torch.allclose(
        future_weights,
        torch.zeros_like(future_weights),
        atol=1e-6,
    )


def test_multi_head_self_attention_weights_sum_to_one() -> None:
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

    _, attn_weights = attention(x, mask=mask)

    row_sums = attn_weights.sum(dim=-1)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    )