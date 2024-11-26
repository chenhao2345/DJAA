import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def cross_entropy_loss(pred_class_outputs, gt_classes, eps, alpha=0.2):
    num_classes = pred_class_outputs.size(1)

    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = F.softmax(pred_class_outputs, dim=1)
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

    log_probs = F.log_softmax(pred_class_outputs, dim=1)
    with torch.no_grad():
        targets = torch.ones_like(log_probs)
        targets *= smooth_param / (num_classes - 1)
        targets.scatter_(1, gt_classes.unsqueeze(1), (1 - smooth_param))

    loss = (-targets * log_probs).sum(dim=1)

    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

    loss = loss.sum() / non_zero_cnt

    return loss


class CircleSoftmax(nn.Module):
    def __init__(self, in_feat, num_classes, s=128, m=0.25):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self.s = s
        self.m = m

    def forward(self, features, weight, targets):
        sim_mat = F.linear(F.normalize(features), F.normalize(weight))
        alpha_p = torch.clamp_min(-sim_mat.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(sim_mat.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        s_p = self.s * alpha_p * (sim_mat - delta_p)
        s_n = self.s * alpha_n * (sim_mat - delta_n)

        targets_onehot = F.one_hot(targets, num_classes=self._num_classes)

        pred_class_logits = targets_onehot * s_p + (1.0 - targets_onehot) * s_n

        loss = cross_entropy_loss(pred_class_logits,targets, eps=0.1)

        return loss

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self.s, self.m
        )