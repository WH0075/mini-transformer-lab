import pytest
import torch

from src.minitransformer.model import TinyTransformerLanguageModel


def test_model_output_shape() -> None:
    batch_size = 2
    seq_len = 8
    vocab_size = 30
    block_size = 8
    embed_dim = 32
    num_heads = 4
    num_layers = 2
    feedforward_dim = 128

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    model = TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
    )

    logits, all_attn_weights = model(input_ids)

    assert logits.shape == torch.Size([batch_size, seq_len, vocab_size])
    assert len(all_attn_weights) == num_layers


def test_model_attention_weight_shapes() -> None:
    batch_size = 2
    seq_len = 8
    vocab_size = 30
    block_size = 8
    embed_dim = 32
    num_heads = 4
    num_layers = 3
    feedforward_dim = 128

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    model = TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
    )

    _, all_attn_weights = model(input_ids)

    assert len(all_attn_weights) == num_layers

    for attn_weights in all_attn_weights:
        assert attn_weights.shape == torch.Size(
            [batch_size, num_heads, seq_len, seq_len]
        )


def test_model_causal_mask_works() -> None:
    batch_size = 2
    seq_len = 8
    vocab_size = 30
    block_size = 8
    embed_dim = 32
    num_heads = 4
    num_layers = 2
    feedforward_dim = 128

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    model = TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
    )

    _, all_attn_weights = model(input_ids)

    for attn_weights in all_attn_weights:
        future_weights = torch.triu(attn_weights, diagonal=1)

        assert torch.allclose(
            future_weights,
            torch.zeros_like(future_weights),
            atol=1e-6,
        )


def test_model_attention_weights_sum_to_one() -> None:
    batch_size = 2
    seq_len = 8
    vocab_size = 30
    block_size = 8
    embed_dim = 32
    num_heads = 4
    num_layers = 2
    feedforward_dim = 128

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    model = TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
    )

    _, all_attn_weights = model(input_ids)

    for attn_weights in all_attn_weights:
        row_sums = attn_weights.sum(dim=-1)

        assert torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-6,
        )


def test_model_rejects_invalid_vocab_size() -> None:
    with pytest.raises(ValueError):
        TinyTransformerLanguageModel(
            vocab_size=0,
            block_size=8,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            feedforward_dim=128,
        )


def test_model_rejects_invalid_block_size() -> None:
    with pytest.raises(ValueError):
        TinyTransformerLanguageModel(
            vocab_size=30,
            block_size=0,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            feedforward_dim=128,
        )


def test_model_rejects_invalid_embed_dim() -> None:
    with pytest.raises(ValueError):
        TinyTransformerLanguageModel(
            vocab_size=30,
            block_size=8,
            embed_dim=0,
            num_heads=4,
            num_layers=2,
            feedforward_dim=128,
        )


def test_model_rejects_invalid_num_heads() -> None:
    with pytest.raises(ValueError):
        TinyTransformerLanguageModel(
            vocab_size=30,
            block_size=8,
            embed_dim=32,
            num_heads=0,
            num_layers=2,
            feedforward_dim=128,
        )


def test_model_rejects_invalid_num_layers() -> None:
    with pytest.raises(ValueError):
        TinyTransformerLanguageModel(
            vocab_size=30,
            block_size=8,
            embed_dim=32,
            num_heads=4,
            num_layers=0,
            feedforward_dim=128,
        )


def test_model_rejects_invalid_feedforward_dim() -> None:
    with pytest.raises(ValueError):
        TinyTransformerLanguageModel(
            vocab_size=30,
            block_size=8,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            feedforward_dim=0,
        )


def test_model_rejects_too_long_sequence() -> None:
    batch_size = 2
    seq_len = 10
    vocab_size = 30
    block_size = 8

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    model = TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        feedforward_dim=128,
    )

    with pytest.raises(ValueError):
        model(input_ids)


def test_model_rejects_invalid_input_shape() -> None:
    vocab_size = 30

    input_ids = torch.randint(0, vocab_size, (8,))

    model = TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=8,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        feedforward_dim=128,
    )

    with pytest.raises(ValueError):
        model(input_ids)