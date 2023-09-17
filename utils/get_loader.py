from configs.config import config
import json
from torch.utils.data import DataLoader
from utils.dataset import Deepfake_Dataset


def get_dataset():
    print('\nLoad Train Data')
    train_real_json = open('../data_label/' + config.real_label_path)
    train_real_dict = json.load(train_real_json)
    print("Train data frames (real): ", len(train_real_dict))

    if config.singleside:
        group_bs = int(config.batch_size / len(config.fake_label_path))
        train_fake_dataloader_list = []
        for fake_dataset in config.fake_label_path:
            with open('../data_label/' + fake_dataset) as f:
                train_fake_dict = json.load(f)
                train_fake_dataloader = DataLoader(Deepfake_Dataset(train_fake_dict), batch_size=group_bs, shuffle=True)
                train_fake_dataloader_list.append(train_fake_dataloader)
        print("Load multiple fake datasets.")
    else:
        train_fake_dict = []
        for fake_dataset in config.fake_label_path:
            with open('../data_label/' + fake_dataset) as f:
                train_fake_dict += json.load(f)
        print("Train data frames (fake): ", len(train_fake_dict))

    test_dataloader_list = []
    print('\nLoad Test Data')
    test_json = open('../data_label/' + config.val_label_path)
    test_dict = json.load(test_json)
    test_dataloader = DataLoader(Deepfake_Dataset(test_dict), batch_size=config.batch_size, shuffle=False)
    test_dataloader_list.append(test_dataloader)

    train_real_dataloader = DataLoader(Deepfake_Dataset(train_real_dict), batch_size=config.batch_size, shuffle=True)
    train_fake_dataloader = DataLoader(Deepfake_Dataset(train_fake_dict), batch_size=config.batch_size, shuffle=True)
    
    if config.singleside:
        return train_real_dataloader, train_fake_dataloader_list, test_dataloader_list
    else:
        return train_real_dataloader, train_fake_dataloader, test_dataloader_list