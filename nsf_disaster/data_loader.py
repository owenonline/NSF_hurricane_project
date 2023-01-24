import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image

# read csv files
train_csv = 'crisis_vision_benchmarks/' \
            'tasks/disaster_types/consolidated/consolidated_disaster_types_train_final.tsv'

test_csv = 'crisis_vision_benchmarks/' \
            'tasks/disaster_types/consolidated/consolidated_disaster_types_test_final.tsv'

dev_csv = 'crisis_vision_benchmarks/' \
            'tasks/disaster_types/consolidated/consolidated_disaster_types_dev_final.tsv'

phenotype = {
    'train': train_csv,
    'test': test_csv,
    'dev': dev_csv,
}

# mapping label to id
label_2_id = {
    'not_disaster': 0,
    'earthquake': 1,
    'fire': 2,
    'flood': 3,
    'hurricane': 4,
    'landslide': 5,
    'other_disaster': 6,
}

class DisasterDataset(Dataset):
    def __init__(self, split='train'):
        self.phenotype = pd.read_csv(
            phenotype[split],
            sep='\t',
        )
        # import pdb; pdb.set_trace()
        self.transforms = transforms.Compose(
            [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            ]
        )

        self.normalize_ = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        self._mean = torch.Tensor([0.485, 0.456, 0.406])
        self._mean = self._mean.reshape((-1,1,1))
        self._std = torch.Tensor([0.229, 0.224, 0.225])
        self._std = self._std.reshape((-1,1,1))

    def __len__(self):
        return int(self.phenotype.shape[0])

    def __getitem__(self, idx):
        img_path = self.phenotype['image_path'].iloc[idx]
        image = Image.open('crisis_vision_benchmarks/{}'.format(img_path))
        image = self.transforms(image)
        image = image.float()
        image = image / 255.0
        image = image - self._mean
        image = image / self._std

        type = self.phenotype['class_label'].iloc[idx]
        type = label_2_id[type]
        return image, type #torch.from_numpy([type]).squeeze().long()
