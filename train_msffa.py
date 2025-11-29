import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from msffa_module import MSFFA_RestorationSubnet
from dataset_msffa import FogRestoreDataset


def train_msffa(
    fog_dir="data/foggy",
    clear_dir="data/clear",
    epochs=10,
    batch_size=4,
    lr=1e-3,
    image_size=(256, 256),
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset + DataLoader
    dataset = FogRestoreDataset(fog_dir, clear_dir, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model
    model = MSFFA_RestorationSubnet().to(device)

    # Loss: Smooth L1 (Huber) like in the paper (Eq. 11–12). :contentReference[oaicite:3]{index=3}
    criterion = nn.SmoothL1Loss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for fog, clear in dataloader:
            fog = fog.to(device)
            clear = clear.to(device)

            # Forward
            restored = model(fog)

            # Loss
            loss = criterion(restored, clear)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * fog.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f}")

        # Save checkpoint occasionally
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/msffa_epoch{epoch}.pth")

    # Final save
    torch.save(model.state_dict(), "msffa_trained.pth")
    print("Training complete. Saved final model to msffa_trained.pth")


if __name__ == "__main__":
    train_msffa()
