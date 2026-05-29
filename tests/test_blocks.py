import pytest
import torch

from src.minitransformer.attention import create_causal_mask
from src.minitransformer.blocks import FeedForward, TransformerBlock


def test_feed_forward_output_shape() -> None:
    batch_size = 2
    seq_len = 4
    embed_dim = 32
    feedforward_dim = 128

    x = torch.randn(batch_size, seq_len, embed_dim)

    feed_forward = FeedForward(
        embed_dim=embed_dim,
        feedforward_dim=feedforward_dim,
    )

    output = feed_forward(x)

    assert output.shape == torch.Size([batch_size, seq_len, embed_dim])


def test_feed_forward_rejects_invalid_embed_dim() -> None:
    with pytest.raises(ValueError):
        FeedForward(embed_dim=0, feedforward_dim=128)


def test_feed_forward_rejects_invalid_feedforward_dim() -> None:
    with pytest.raises(ValueError):
        FeedForward(embed_dim=32, feedforward_dim=0)


def test_transformer_block_output_shape() -> None:
    batch_size = 2
    seq_len = 4
    embed_dim = 32
    num_heads = 4
    feedforward_dim = 128

    x = torch.randn(batch_size, seq_len, embed_dim)

    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
    )

    output, attn_weights = block(x)

    assert output.shape == torch.Size([batch_size, seq_len, embed_dim])
    assert attn_weights.shape == torch.Size(
        [batch_size, num_heads, seq_len, seq_len]
    )


def test_transformer_block_with_causal_mask() -> None:
    batch_size = 2
    seq_len = 4
    embed_dim = 32
    num_heads = 4
    feedforward_dim = 128

    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = create_causal_mask(seq_len)

    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
    )

    output, attn_weights = block(x, mask=mask)

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


def test_transformer_block_attention_weights_sum_to_one() -> None:
    batch_size = 2
    seq_len = 4
    embed_dim = 32
    num_heads = 4
    feedforward_dim = 128

    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = create_causal_mask(seq_len)

    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
    )

    _, attn_weights = block(x, mask=mask)

    row_sums = attn_weights.sum(dim=-1)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    )


def test_transformer_block_rejects_invalid_embed_dim() -> None:
    with pytest.raises(ValueError):
        TransformerBlock(
            embed_dim=0,
            num_heads=4,
            feedforward_dim=128,
        )


def test_transformer_block_rejects_invalid_num_heads() -> None:
    with pytest.raises(ValueError):
        TransformerBlock(
            embed_dim=32,
            num_heads=0,
            feedforward_dim=128,
        )


def test_transformer_block_rejects_invalid_feedforward_dim() -> None:
    with pytest.raises(ValueError):
        TransformerBlock(
            embed_dim=32,
            num_heads=4,
            feedforward_dim=0,
        )


def test_transformer_block_rejects_invalid_input_dim() -> None:
    batch_size = 2
    seq_len = 4
    wrong_embed_dim = 16

    x = torch.randn(batch_size, seq_len, wrong_embed_dim)

    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
        feedforward_dim=128,
    )

    with pytest.raises(ValueError):
        block(x)