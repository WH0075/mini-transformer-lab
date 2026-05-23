import torch


def main() -> None:
    print("PyTorch version:", torch.__version__)

    print("\n1. Create tensors")
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print("a =")
    print(a)
    print("a.shape:", a.shape)
    print("a.dtype:", a.dtype)
    print("a.device:", a.device)

    print("\n2. Random / zeros / ones")
    x = torch.randn(2, 3)
    z = torch.zeros(2, 3)
    o = torch.ones(2, 3)

    print("x =")
    print(x)
    print("x.shape:", x.shape)
    print("zeros shape:", z.shape)
    print("ones shape:", o.shape)

    print("\n3. Matrix multiplication")
    m1 = torch.randn(2, 3)
    m2 = torch.randn(3, 4)
    y = m1 @ m2

    print("m1.shape:", m1.shape)
    print("m2.shape:", m2.shape)
    print("y.shape:", y.shape)

    assert y.shape == (2, 4)

    print("\n4. Broadcasting")
    logits = torch.randn(2, 4)
    bias = torch.randn(4)
    output = logits + bias

    print("logits.shape:", logits.shape)
    print("bias.shape:", bias.shape)
    print("output.shape:", output.shape)

    assert output.shape == (2, 4)

    print("\n5. Batch dimension")
    batch_images = torch.randn(8, 3, 32, 32)

    print("batch_images.shape:", batch_images.shape)
    print("8 means batch size")
    print("3 means channels")
    print("32 and 32 mean height and width")

    print("\nAll tensor basic checks passed!")


if __name__ == "__main__":
    main()