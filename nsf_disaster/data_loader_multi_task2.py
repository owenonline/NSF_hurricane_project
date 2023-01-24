import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image
import glob
import os
from PIL import ImageFile
from sklearn.utils.class_weight import compute_class_weight
from torchvision.io import read_image

ImageFile.LOAD_TRUNCATED_IMAGES = True

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
# read csv files
# train_csv_type = 'crisis_vision_benchmarks/' \
#             'tasks/disaster_types/consolidated/consolidated_disaster_types_train_final.tsv'
# test_csv_type = 'crisis_vision_benchmarks/' \
#             'tasks/disaster_types/consolidated/consolidated_disaster_types_test_final.tsv'
# dev_csv_type = 'crisis_vision_benchmarks/' \
#             'tasks/disaster_types/consolidated/consolidated_disaster_types_dev_final.tsv'

# train_csv_damage = 'crisis_vision_benchmarks/' \
#             'tasks/damage_severity/consolidated/consolidated_damage_train_final.tsv'
# test_csv_damage = 'crisis_vision_benchmarks/' \
#             'tasks/damage_severity/consolidated/consolidated_damage_test_final.tsv'
# dev_csv_damage = 'crisis_vision_benchmarks/' \
#             'tasks/damage_severity/consolidated/consolidated_damage_dev_final.tsv'

# train_csv_humanitarian = 'crisis_vision_benchmarks/' \
#             'tasks/humanitarian/consolidated/consolidated_hum_train_final.tsv'
# test_csv_humanitarian = 'crisis_vision_benchmarks/' \
#             'tasks/humanitarian/consolidated/consolidated_hum_test_final.tsv'
# dev_csv_humanitarian = 'crisis_vision_benchmarks/' \
#             'tasks/humanitarian/consolidated/consolidated_hum_dev_final.tsv'

# train_csv_informative = 'crisis_vision_benchmarks/' \
#             'tasks/informative/consolidated/consolidated_info_train_final.tsv'
# test_csv_informative = 'crisis_vision_benchmarks/' \
#             'tasks/informative/consolidated/consolidated_info_test_final.tsv'
# dev_csv_informative = 'crisis_vision_benchmarks/' \
#             'tasks/informative/consolidated/consolidated_info_dev_final.tsv'

# phenotype = {
#     'train': train_csv,
#     'test': test_csv,
#     'dev': dev_csv,
# }

# mapping label to id


class DisasterDataset(Dataset):
    def __init__(self, task = 'disaster_types', split='train'):
        # self.phenotype = pd.read_csv( 
        #     phenotype[split],
        #     sep='\t',
        # )
        # import pdb; pdb.set_trace()
        self.task = task
        phenotypes = glob.glob('./crisis_vision_benchmarks/tasks/'+task+'/consolidated/*'+split+'_final.tsv')
        self.phenotypes = []
        for phenotype in phenotypes:
            temp = pd.read_csv(phenotype,sep='\t',)
            # import pdb; pdb.set_trace()
            idxes = []
            for idx in range(temp.shape[0]):
                img_dir = temp['image_path'].iloc[idx]
                img_dir = img_dir.replace('/export/sc2/aidr/experiments/exp_crisisdps_image/','')
                if not os.path.exists('crisis_vision_benchmarks/{}'.format(img_dir)):
                    # import pdb; pdb.set_trace()
                    print(phenotype,img_dir)
                    idxes.append(idx)
            if not idxes:
                temp.drop(idxes)                
            self.phenotypes.append( temp)
        self.phenotypes = pd.concat(self.phenotypes,axis=0)
        self.label_2_id = { 'disaster_types':{
                        'not_disaster': 0,
                        'earthquake': 1,
                        'fire': 2,
                        'flood': 3,
                        'hurricane': 4,
                        'landslide': 5,
                        'other_disaster': 6,
                    },
                    'damage_severity':{'little_or_none':0,'mild':1,'severe':2},
                    'humanitarian':{'affected_injured_or_dead_people':0,'infrastructure_and_utility_damage':1,'not_humanitarian':2,'rescue_volunteering_or_donation_effort':3},
                    'informative':{'not_informative':1,'informative':0}
                    }
        
        # import pdb; pdb.set_trace()
        class_label = self.phenotypes['class_label'].to_numpy()
        self.num_classes = np.unique(class_label)
        self.class_weights=compute_class_weight(class_weight='balanced',classes=self.num_classes,y=class_label)
        # normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if split == 'train':
        

            self.transforms= transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transforms = transforms.Compose(
            [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ]
        )

        
        self._mean = torch.Tensor([0.485, 0.456, 0.406])
        self._mean = self._mean.reshape((-1,1,1))
        self._std = torch.Tensor([0.229, 0.224, 0.225])
        self._std = self._std.reshape((-1,1,1))

    def __len__(self):
        return int(self.phenotypes.shape[0])

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        img_path = self.phenotypes['image_path'].iloc[idx]
        img_path = img_path.replace('/export/sc2/aidr/experiments/exp_crisisdps_image/','')
        # try:
        # image = Image.open('crisis_vision_benchmarks/{}'.format(img_path)) 
        image = pil_loader('crisis_vision_benchmarks/{}'.format(img_path))
        # print(img_path)
        # print(image.size)
        # print(image.shape)
        # image = image.convert('RGB')
        # image = read_image('crisis_vision_benchmarks/{}'.format(img_path))
        # # except:
            # print(img_path)
        
        image = self.transforms(image)
        # print(image)
        # image = image.float()
        # image = image / 255.0
        # try:
        # if image.shape[0] >3:
        #     image = image[:3,:]
        
        # image = image - self._mean
        # except:
        # print(image.shape)
        # print(self._mean.shape)
        # image = image / self._std

        type = self.phenotypes['class_label'].iloc[idx]
        type = self.label_2_id[self.task][type]
        return image, type #torch.from_numpy([type]).squeeze().long()
