import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class WrapData(Dataset):
    def __init__(self, items, image_path, map_path, transform):
        super(WrapData, self).__init__()
        self.image_path = image_path
        self.map_path = map_path
        self.items = items
        self.transform = transform
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_path, self.items[index]))
        map_image = torch.load(os.path.join(self.map_path, self.items[index].replace('.jpg', '.pt'))).to(dtype=torch.float32)
        return self.transform(image), map_image.unsqueeze(0)
