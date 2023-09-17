from utils.utils import AverageMeter, accuracy, Logger
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from datetime import datetime
from configs.config import config
from sklearn.metrics import roc_auc_score

global_prob_list = []
global_label_list = []

def eval_state(probs, labels, thr):
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP

def calculate(probs, labels):
    TN, FN, FP, TP = eval_state(probs, labels, 0.5)
    P = 2.0 if (TP + FP == 0) else TP / float(TP + FP) # 2.0 denotes not exist
    R = 2.0 if (TP + FN == 0) else TP / float(TP + FN)
    return P, R

def val_one_dataset(valid_dataloader, model, name):
    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()
    model.eval()
    total = 0
    correct = 0

    global global_label_list
    global global_prob_list
    prob_list = []
    label_list = []

    with torch.no_grad():
        for input, target in valid_dataloader:
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            feature, cls_out = model(input)

            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            _, predicted = torch.max(cls_out.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()

            prob_list = np.append(prob_list, prob)
            label_list = np.append(label_list, label)
            global_prob_list = np.append(global_prob_list, prob)
            global_label_list = np.append(global_label_list, label)
            loss = criterion(cls_out, target)
            acc_valid = accuracy(cls_out, target, topk=(1,))
            valid_losses.update(loss.item())
            valid_top1.update(acc_valid[0])

    Precision, Recall = calculate(prob_list, label_list)
    auc_score = roc_auc_score(label_list, prob_list)

    return valid_losses.avg, valid_top1.avg.cpu().data.numpy()[0], total, correct, Precision, Recall, auc_score

def val_full(dataloader_list, model, epoch):

    # This supports the evaluation of model on multiple test/validation sets.
    
    loss_list = []
    acc_list = []
    precision_list = []
    recall_list = []
    auc_list = []
    full_total = 0
    full_correct = 0
    log = Logger()
    log.open('./logs/' + config.log_name + '_test.txt', mode='a')
    log.write('\n========================= Test =========================', is_terminal=0)
    log.write("\n%s Epoch: %d \n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch), is_terminal=0)
    for i in range(len(dataloader_list)):
        dataset_name = config.val_label_path[i]
        loss, acc, total, correct, precision, recall, auc_score = val_one_dataset(dataloader_list[i], model, dataset_name)
        log.write('\nacc test (' + dataset_name + '): %6.4f' % (acc), is_terminal=0)
        log.write('\nloss test (' + dataset_name + '): %6.4f' % (loss), is_terminal=0)
        log.write('\nprecision test (' + dataset_name + '): %6.4f' % (precision), is_terminal=0)
        log.write('\nrecall test (' + dataset_name + '): %6.4f' % (recall), is_terminal=0)
        log.write('\nAUC test (' + dataset_name + '): %6.4f' % (auc_score), is_terminal=0)
        log.write('\n', is_terminal=0)
        loss_list.append(loss)
        acc_list.append(acc)
        auc_list.append(auc_score)
        precision_list.append(precision) 
        recall_list.append(recall)
        full_correct += correct
        full_total += total
    
    if len(dataloader_list) > 1:
        global global_label_list
        global global_prob_list
        global_label_list = []
        global_prob_list = []
        full_correct = full_correct.cpu().data.numpy().tolist()
        total_acc = round(full_correct / full_total, 5)
        loss_list.append(np.sum(loss_list) / len(loss_list))
        acc_list.append(total_acc*100)
        total_precision, total_recall = calculate(global_prob_list, global_label_list)
        total_auc = roc_auc_score(global_label_list, global_prob_list)
        precision_list.append(total_precision)
        recall_list.append(total_recall)
        auc_list.append(total_auc)
        log.write('\n--------------------------------\n', is_terminal=0)
        log.write('\nacc test (full dataset): %6.4f' % (acc_list[-1]), is_terminal=0)
        log.write('\nloss test (full dataset): %6.4f' % (loss_list[-1]), is_terminal=0)
        log.write('\nprecision test (full dataset): %6.4f' % (precision_list[-1]), is_terminal=0)
        log.write('\nrecall test (full dataset): %6.4f' % (recall_list[-1]), is_terminal=0)
        log.write('\nAUC test (full dataset): %6.4f' % (auc_list[-1]), is_terminal=0)
        # log.write('\nInconsistency rate (full dataset): %6.3f' % inconsistency, is_terminal=0)
        log.write('\n', is_terminal=0)
    log.write('\n========================= End =========================\n', is_terminal=0)
    return loss_list, acc_list, precision_list, recall_list, auc_list

