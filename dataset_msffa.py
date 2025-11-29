import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class FogRestoreDataset(Dataset):
    """
    Dataset for foggy -> clear image restoration.
    Expects two folders: foggy/ and clear/ with matching filenames.
    """
    def __init__(self, fog_dir, clear_dir, image_size=(256, 256)):
        self.fog_dir = fog_dir
        self.clear_dir = clear_dir
        self.image_size = image_size

        self.filenames = sorted(os.listdir(fog_dir))

        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),          # [0,1]
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        fog_path = os.path.join(self.fog_dir, fname)
        clear_path = os.path.join(self.clear_dir, fname)

        fog = Image.open(fog_path).convert("RGB")
        clear = Image.open(clear_path).convert("RGB")

        fog = self.transform(fog)
        clear = self.transform(clear)

        return fog, clear
