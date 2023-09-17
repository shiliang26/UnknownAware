import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from models.xception import xception

def get_base_model(backbone):
    if backbone == 'xception':
        model = xception(pretrained = True)
        model.feature_size = 2048
    else:
        print(backbone, "Model undefined.")
        exit(0)
    return model


class base_model(nn.Module):
    def __init__(self, backbone='xception'):
        super(base_model, self).__init__()
        self.backbone = get_base_model(backbone)
        self.classifier = nn.Linear(self.backbone.feature_size, 2)

    def forward(self, input, gt=None, mode='test'):
        feature = self.backbone(input)
        output = self.classifier(feature)
        return feature, output


class rsc_model(nn.Module):

    # Adapted from GitHub repository DomainBed https://github.com/facebookresearch/DomainBed

    def __init__(self, backbone='xception'):
        super(rsc_model, self).__init__()
        self.backbone = get_base_model(backbone)
        self.classifier = nn.Linear(self.backbone.feature_size, 2)
        self.drop_f = (1 - 1 / 3) * 100
        self.drop_b = (1 - 1 / 3) * 100
 
    def forward(self, input, gt=None, mode='test'):

        input_ = Variable(input, requires_grad=True)
        all_f = self.backbone(input_)
        all_f_full = all_f
        
        if mode == 'train':

            all_o = torch.nn.functional.one_hot(gt, 2)
            all_p = self.classifier(all_f)

            all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

            percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
            percentiles = torch.Tensor(percentiles)
            percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
            mask_f = all_g.lt(percentiles.to('cuda')).float()

            all_f_muted = all_f * mask_f
            all_p_muted = self.classifier(all_f_muted)

            all_s = F.softmax(all_p, dim=1)
            all_s_muted = F.softmax(all_p_muted, dim=1)
            changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
            percentile = np.percentile(changes.detach().cpu(), self.drop_b)
            mask_b = changes.lt(percentile).float().view(-1, 1)
            mask = (mask_f + mask_b) > 0
            
            all_f = all_f * mask

        output = self.classifier(all_f)

        return all_f_full, output
