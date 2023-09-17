from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, time_to_str
from utils.validate import val_full
from utils.get_loader import get_dataset
import models.gadgets as gadgets
import random
import numpy as np
from configs.config import config
from datetime import datetime
import time
from timeit import default_timer as timer

import os
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses, miners

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = 'cuda'

def train():

    assert config.singleside == True
    
    if config.rsc:
        mode = 'rsc'
    else:
        mode = None

    mkdirs()

    train_real_dataloader, train_fake_dataloader_list, test_dataloader = get_dataset()
    num_fake_datasets = len(train_fake_dataloader_list)

    is_best_AUC = True
    is_best_ACC = True
    valid_args = [np.inf, 0, 0, 0, 0, 0]

    best_AUC_ACC = 0.0
    best_AUC_loss = 0.0
    best_auc = 0.0
    best_acc = 0.0
    best_ACC_loss = 0.0
    best_ACC_AUC = 0.0

    loss_classifier = AverageMeter()
    classifer_top1 = AverageMeter()

    if config.rsc:
        net = gadgets.rsc_model(config.model)
    else:
        net = gadgets.base_model(config.model)
    net = net.cuda()
    
    log = Logger()
    log.open(config.logs + 'baseline_' + config.log_name +'.txt', mode='a')
    log.write("\n-------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 45))
    log.write('Random seed: %d\n' % config.seed)
    log.write('Comment: %s\n' % config.comment)
    log.write('** start training target model! **\n')
    log.write(
        '--------|---------- VALID ----------|------- Train --------|----------Best AUC ---------|--------------|\n')
    log.write(
        '  iter  |   loss    top-1    AUC    |     loss    top-1    |     loss   top-1    AUC    |    time      |\n')
    log.write(
        '-------------------------------------------------------------------------------------------------------|\n')
    start = timer()
    criterion = {
        'softmax': nn.CrossEntropyLoss().cuda(),
        'triplet': losses.TripletMarginLoss().cuda(),
        'miner': miners.MultiSimilarityMiner().cuda()
    }

    optimizer = optim.Adam(net.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

    iter_per_epoch = config.iter_per_epoch # iters that the model need to be tested

    train_real_iter = iter(train_real_dataloader)
    train_real_iters_per_epoch = len(train_real_iter)

    train_fake_iters = [iter(dataloader) for dataloader in train_fake_dataloader_list]
    train_fake_iters_per_epoch = [len(fake_iter) for fake_iter in train_fake_iters]

    max_iter = config.max_iter
    epoch = 0

    if(len(config.gpus) > 1):
        net = torch.nn.DataParallel(net).cuda()

    for iter_num in range(max_iter+1):

        if (iter_num % train_real_iters_per_epoch == 0):
            train_real_iter = iter(train_real_dataloader)
        for i in range(num_fake_datasets):
            if (iter_num % train_fake_iters_per_epoch[i] == 0):
                train_fake_iters[i] = iter(train_fake_dataloader_list[i])
        if (iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1
            loss_classifier.reset()
            classifer_top1.reset()

        

        net.train(True)

        optimizer.zero_grad()

        img_real, label_real = train_real_iter.next()
        img_real = img_real.cuda()
        label_real = label_real.cuda()

        img_label_pairs = [train_fake_iter.next() for train_fake_iter in train_fake_iters]
        img_fake = torch.cat([img for (img, label) in img_label_pairs]).cuda()
        label_fake = torch.cat([label for (img, label) in img_label_pairs]).cuda()

        input_data = torch.cat([img_real, img_fake], dim=0)
        input_label = torch.cat([label_real, label_fake], dim=0)

        real_domain_label = torch.LongTensor(label_real.shape).fill_(0).cuda()
        fake_domain_labels = []
        
        for i in range(len(img_fake)):
            # Use the shape of current batch.
            fake_domain_labels.append(torch.LongTensor(1).fill_(i + 1).cuda())
        fake_domain_label = torch.cat([fake_domain_labels[i] for i in range(len(img_fake))])
        domain_label = torch.cat([real_domain_label, fake_domain_label])
    
        feature, classifier_out = net(input_data, input_label, mode='train')
        
        cls_loss = criterion["softmax"](classifier_out, input_label)
        if config.singleside:
            hard_pairs = criterion['miner'](feature, domain_label)
            cls_loss += criterion["triplet"](feature, domain_label, hard_pairs)
            cls_loss += torch.norm(feature[:len(img_real)]) / 2048

        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()
        
        loss_classifier.update(cls_loss.item())
        acc = accuracy(classifier_out, input_label, topk=(1,))

        classifer_top1.update(acc[0])
        print('\r', end='', flush=True)
        print(
            '  %4.1f  |  %6.4f  %6.4f  %6.4f  |   %6.4f   %6.4f   |   %6.4f  %6.4f  %6.4f  |    %s'
            % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[1], valid_args[4],
                loss_classifier.avg, classifer_top1.avg,
                round(best_AUC_loss, 5), round(best_AUC_ACC, 5), round(best_auc, 5),
                time_to_str(timer() - start, 'min'))
            , end='', flush=True)
            
        
        if (iter_num != 0 and (iter_num+1) % iter_per_epoch == 0):

            loss_list, acc_list, precision_list, recall_list, auc_list = val_full(test_dataloader, net, epoch)
            
            # 0:loss, 1:top-1
            valid_args[0] = loss_list[-1]
            valid_args[1] = acc_list[-1]
            valid_args[2] = precision_list[-1]
            valid_args[3] = recall_list[-1]
            valid_args[4] = auc_list[-1]
            # judge model according to ACC              # 2021.1.13 changed this to best AUC
            is_best_AUC = valid_args[4] >= best_auc

            if is_best_AUC:
                best_AUC_ACC = valid_args[1]        # This means the accuracy of the model with best AUC
                best_AUC_loss = valid_args[0]
                best_auc = valid_args[4]

            if epoch % config.save_freq == 0 or is_best_AUC:
                save_list = [epoch, valid_args, best_auc]
                save_checkpoint(save_list, is_best_AUC, net)

            print('\r', end='', flush=True)
            log.write(
                '  %4.1f  |  %6.4f  %6.4f  %6.4f  |   %6.4f   %6.4f   |   %6.4f  %6.4f  %6.4f  |    %s'
                % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[1], valid_args[4],
                loss_classifier.avg, classifer_top1.avg,
                round(best_AUC_loss, 5), round(best_AUC_ACC, 5), round(best_auc, 5), 
                time_to_str(timer() - start, 'min')))
            log.write('\n')
            time.sleep(0.01)

        
        
if __name__ == '__main__':
    train()
