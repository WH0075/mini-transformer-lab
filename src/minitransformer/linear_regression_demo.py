import torch


def main():
    torch.manual_seed(32)

    x = torch.linspace(-2, 2, steps=100).reshape(-1, 1)
    noise = 0.1 * torch.randn_like(x)
    y = 3 * x + 2 + noise

    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    learning_rate = 0.1
    num_epochs = 100

    print(f"Initial w: {w.item(): .4f}, b: {b.item(): .4f}")
    print(f"x.shape: {x.shape}")
    print(f"y.shapa: {y.shape}")

    for epoch in range(num_epochs):
        y_pred = w * x + b

        loss = ((y_pred - y) ** 2).mean()
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

            w.grad.zero_()
            b.grad.zero_()
        
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1:03d}/{num_epochs}], "
                f"loss: {loss.item():.6f}, "
                f"w: {w.item():.4f}, "
                f"b: {b.item():.4f}"
            )
    
    print("\nTraining finished.")
    print(f"Learned w: {w.item():.4f}")
    print(f"Learned b: {b.item():.4f}")

if __name__ == "__main__":
    main()