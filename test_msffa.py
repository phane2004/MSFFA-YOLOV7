import torch
from msffa_module import MSFFA_RestorationSubnet


def main():
    # Create a dummy foggy image batch: (batch=1, channels=3, height=256, width=256)
    x = torch.randn(1, 3, 256, 256)

    model = MSFFA_RestorationSubnet()
    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()
