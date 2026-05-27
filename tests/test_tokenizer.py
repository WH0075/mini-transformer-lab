from src.minitransformer.tokenizer import CharTokenizer


def test_char_tokenizer_encode_decode() -> None:
    text = "hello world"
    tokenizer = CharTokenizer(text)

    sample_text = "hello"
    encoded_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(encoded_ids)

    assert decoded_text == sample_text


def test_char_tokenizer_vocab_size() -> None:
    text = "hello"
    tokenizer = CharTokenizer(text)

    assert tokenizer.volab_size == 4
    assert set(tokenizer.chars) == {"h", "e", "l", "o"}


def test_char_tokenizer_stoi_itos_are_consistent() -> None:
    text = "abc"
    tokenizer = CharTokenizer(text)

    for ch in text:
        idx = tokenizer.stoi[ch]
        assert tokenizer.itos[idx] == ch