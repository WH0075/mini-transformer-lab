from pathlib import Path


class CharTokenizer:
    def __init__(self, text: str) -> None:
        if not text:
            raise ValueError("Texe corpus must not be empty.")
        
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: idx for idx, ch in enumerate(self.chars)}
        self.itos = {idx: ch for idx, ch in enumerate(self.chars)}
        self.volab_size = len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]
    
    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[idx] for idx in ids)


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    return text


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    corpus_path = project_root / "data" / "raw" / "tiny_corpus.txt"

    text = load_text(corpus_path)
    tokenizer = CharTokenizer(text)

    sample_text = "hello transformer"
    encoded_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(encoded_ids)

    print(f"Corpus path: {corpus_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Characters: {tokenizer.chars}")
    print(f"stoi: {tokenizer.stoi}")
    print(f"itos: {tokenizer.itos}")
    print()
    print(f"Original text: {sample_text}")
    print(f"Encoded ids: {encoded_ids}")
    print(f"Decoded text: {decoded_text}")

    assert decoded_text == sample_text

    print("\nTokenizer check passed!")


if __name__ == "__main__":
    main()
