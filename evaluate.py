
from sklearn.metrics import roc_auc_score
import models.gadgets as gadgets
import random
import numpy as np
from configs.config import config

import os
import cv2
import torch
from torch.nn import functional as F

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2 

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = 'cuda'

# Ugly but straightforward testing.

transform = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),ToTensorV2()])

test_path = config.test_label_path
test_json = open('../data_label/' + test_path)
print("Test dataset:", test_path)
samples = json.load(test_json)

image_path, image_label = [sample['path'] for sample in samples], [sample['label'] for sample in samples]
total = len(samples)

net = gadgets.base_model(config.model).cuda()
# net = torch.nn.DataParallel(net).cuda()

modelpath = config.evaluate_model_path

thre = 0.5

checkpoint = torch.load(modelpath)
net.load_state_dict(checkpoint['state_dict'])
net.eval()

pred_list = []
label_list = []

TP, TN, FP, FN = 0, 0, 0, 0

for i, (path, label) in tqdm(enumerate(zip(image_path, image_label))):

    img = cv2.imread(path)
    img_s = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_s = transform(image = img_s)['image']
    img = img_s.unsqueeze(0).cuda()

    feature, output = net(img)
    pred = F.softmax(output, dim=1).cpu().data.numpy()[0][1]

    pred_list.append(pred)
    label_list.append(label)

    if pred > thre:
        if label == 1:
            TP += 1
        else:
            FP += 1
    else:
        if label == 1:
            FN += 1
        else:
            TN += 1

auc = roc_auc_score(label_list, pred_list)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
acc = (TP + TN) / total

print('Model: ', os.path.basename(modelpath))
print('ACC:', round(acc,4))
print('AUC:', round(auc,4))

f = open('logs/evaluate.txt', 'a')
f.write('\nModel: ' + os.path.basename(modelpath))
f.write('\nTest path: ' + test_path)
f.write('\nACC: ' + str(round(acc, 4)))
f.write('\nAUC: ' + str(round(auc, 4)))
f.write('\n')
f.close()
