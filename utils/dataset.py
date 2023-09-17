import cv2
from torch.utils.data import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
sys.path.append("..")
from configs.config import config

import albumentations as A
from albumentations.pytorch import ToTensorV2 

class Deepfake_Dataset(Dataset):
    def __init__(self, data_dict):
        self.remove = []
        self.photo_path = [dicti['path'] for dicti in data_dict]
        self.photo_label = [dicti['label']for dicti in data_dict]
                    
        if not config.aug:
            self.transforms = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
        else:
            self.transforms = A.Compose([
                A.HorizontalFlip(),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                A.GaussNoise(p=0.1),
                A.GaussianBlur(blur_limit=3, p=0.05),
                A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
                A.ToGray(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])


    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):

        img_path = self.photo_path[item]
        label = self.photo_label[item]

        img = cv2.imread(img_path)
        img_s = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_s = self.transforms(image = img_s)['image']
        img = img_s

        return img, label
