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

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train():
    mkdirs()
    train_real_dataloader, train_fake_dataloader, test_dataloader = get_dataset()

    if config.rsc:
        net = gadgets.rsc_model(config.model)
    else:
        net = gadgets.base_model(config.model)
    
    best_auc = 0.0
    valid_args = [np.inf, 0, 0, 0, 0, 0]

    best_AUC_ACC = 0.0
    best_AUC_loss = 0.0
    best_auc = 0.0

    loss_classifier = AverageMeter()
    classifer_top1 = AverageMeter()


    net = net.cuda()
    if config.pretrained_weights is not None:
        checkpoint = torch.load(config.pretrained_weights)
        net.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint['epoch'] + 1
        beginning_iter = checkpoint['epoch'] // config.iter_per_epoch
    else:
        epoch = 0
        beginning_iter = 0

    log = Logger()
    log.open(config.logs + 'baseline_' + config.log_name +'.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    log.write('Random seed: %d\n' % config.seed)
    log.write('Comment: %s\n' % config.comment)
    log.write('Log: %s\n' % config.log_name)
    log.write('** start training target model! **\n')
    log.write(
        '--------|---------- VALID ----------|------- Train --------|----------Best AUC ---------|--------------|\n')
    log.write(
        '  iter  |   loss    top-1    AUC    |     loss    top-1    |     loss   top-1    AUC    |    time      |\n')
    log.write(
        '-------------------------------------------------------------------------------------------------------|\n')
    start = timer()
    criterion = {
        'softmax': nn.CrossEntropyLoss().cuda()
    }

    optimizer = optim.Adam(net.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

    iter_per_epoch = config.iter_per_epoch # iters that the model need to be tested

    max_iter = config.max_iter

    train_real_iter = iter(train_real_dataloader)
    train_real_iters_per_epoch = len(train_real_iter)
    train_fake_iter = iter(train_fake_dataloader)
    train_fake_iters_per_epoch = len(train_fake_iter)

    if(len(config.gpus) > 1):
        net = torch.nn.DataParallel(net).cuda()

    for iter_num in range(beginning_iter, max_iter+1):
        if (iter_num % train_real_iters_per_epoch == 0):
            train_real_iter = iter(train_real_dataloader)
        if (iter_num % train_fake_iters_per_epoch == 0):
            train_fake_iter = iter(train_fake_dataloader)
        if (iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1
            loss_classifier.reset()
            classifer_top1.reset()

        net.train(True)

        optimizer.zero_grad()

        img_real, label_real = train_real_iter.next()
        img_real = img_real.cuda()
        label_real = label_real.cuda()

        img_fake, label_fake = train_fake_iter.next()
        img_fake = img_fake.cuda()
        label_fake = label_fake.cuda()

        input_data = torch.cat([img_real, img_fake], dim=0)
        input_label = torch.cat([label_real, label_fake], dim=0)

        batchindex = list(range(len(input_data)))
        random.shuffle(batchindex)
        input_data_random = input_data[batchindex, :]
        input_label_random = input_label[batchindex]  

        _, classifier_out = net(input_data_random, input_label_random, mode = 'train')
        
        cls_loss = criterion["softmax"](classifier_out, input_label_random)

        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()

        loss_classifier.update(cls_loss.item())
        acc = accuracy(classifier_out, input_label_random, topk=(1,))

        classifer_top1.update(acc[0])
        print('\r', end='', flush=True)
        print(
            '  %4.1f  |  %6.4f  %6.4f  %6.4f  |   %6.4f   %6.4f   |   %6.4f  %6.4f  %6.4f  |    %s'
            % (
                epoch + (iter_num % iter_per_epoch) / iter_per_epoch,
                valid_args[0], valid_args[1], valid_args[4],
                loss_classifier.avg, classifer_top1.avg,
                round(best_AUC_loss, 5), round(best_AUC_ACC, 5), round(best_auc, 5),
                time_to_str(timer() - start, 'min'))
            , end='', flush=True)
                
            
        if (iter_num != 0 and (iter_num+1) % config.iter_per_eval == 0):
            
            loss_list, acc_list, precision_list, recall_list, auc_list = val_full(test_dataloader, net, epoch)
            
            # 0:loss, 1:top-1
            valid_args[0] = loss_list[-1]
            valid_args[1] = acc_list[-1]
            valid_args[2] = precision_list[-1]
            valid_args[3] = recall_list[-1]
            valid_args[4] = auc_list[-1]
            
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
                epoch + (iter_num % iter_per_epoch) / iter_per_epoch,
                valid_args[0], valid_args[1], valid_args[4],
                loss_classifier.avg, classifer_top1.avg,
                round(best_AUC_loss, 5), round(best_AUC_ACC, 5), round(best_auc, 5), 
                time_to_str(timer() - start, 'min')))
            log.write('\n')
            time.sleep(0.01)
        
if __name__ == '__main__':
    train()
