import os
import torch
from PIL import Image
import torchvision.transforms as T
from msffa_module import MSFFA_RestorationSubnet


def load_image(path, size=(256, 256)):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor()
    ])
    return transform(img).unsqueeze(0), img.size  # (1,3,H,W), original size


def save_image(tensor, path, orig_size=None):
    tensor = tensor.clamp(0, 1).squeeze(0)     # (3,H,W)
    img = T.ToPILImage()(tensor)
    if orig_size is not None:
        img = img.resize(orig_size)
    img.save(path)
    print("Saved to:", os.path.abspath(path))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = MSFFA_RestorationSubnet().to(device)
    model.load_state_dict(torch.load("msffa_trained.pth", map_location=device))
    model.eval()

    x, orig_size = load_image("foggy.jpg")
    x = x.to(device)

    with torch.no_grad():
        y = model(x)

    save_image(y.cpu(), "foggy_enhanced_trained.jpg", orig_size)


if __name__ == "__main__":
    main()
