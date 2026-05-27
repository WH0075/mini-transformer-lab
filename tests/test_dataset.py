import pytest
import torch

from src.minitransformer.datasets import CharLanguageModelDataset


def test_char_language_model_dataset_length() -> None:
    token_ids = [0, 1, 2, 3, 4]
    dataset = CharLanguageModelDataset(token_ids=token_ids, block_size=3)

    assert len(dataset) == 2


def test_char_language_model_dataset_item_shape() -> None:
    token_ids = [0, 1, 2, 3, 4]
    dataset = CharLanguageModelDataset(token_ids=token_ids, block_size=3)

    x, y = dataset[0]

    assert x.shape == torch.Size([3])
    assert y.shape == torch.Size([3])
    

def test_char_language_model_dataset_shift_by_one() -> None:
    token_ids = [0, 1, 2, 3, 4]
    dataset = CharLanguageModelDataset(token_ids=token_ids, block_size=3)

    x, y = dataset[0]

    assert x.tolist() == [0, 1, 2]
    assert y.tolist() == [1, 2, 3]


def test_char_language_model_second_item() -> None:
    token_ids = [0, 1, 2, 3, 4]
    dataset = CharLanguageModelDataset(token_ids=token_ids, block_size=3)

    x, y = dataset[1]

    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [2, 3, 4]


def test_char_language_model_dataset_rejects_invalid_block_size() -> None:
    token_ids = [0, 1, 2, 3, 4]

    with pytest.raises(ValueError):
        CharLanguageModelDataset(token_ids=token_ids, block_size=0)


def test_char_language_model_dataset_rejects_too_short_data() -> None:
    token_ids = [0, 1, 2]

    with pytest.raises(ValueError):
        CharLanguageModelDataset(token_ids=token_ids, block_size=3)
