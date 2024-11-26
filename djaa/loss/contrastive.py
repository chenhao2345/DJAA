import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class CenterContrastiveLoss(nn.Module):
    def __init__(self, T=1.0, N_neg=50, num_classes=500):
        super(CenterContrastiveLoss, self).__init__()
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = CrossEntropyLabelSmooth(num_classes=num_classes)
        self.T = T
        self.N_neg = N_neg

    def forward(self, f, centers, label):
        batchSize = f.shape[0]
        mat = torch.matmul(f, centers.transpose(0, 1))
        positives = []
        negatives = []
        all_label = torch.arange(centers.shape[0]).long()

        for i in range(batchSize):
            pos = mat[i, label[i]]
            neg = mat[i, all_label!=label[i]]
            neg, _ = torch.topk(neg, k=self.N_neg, largest=True)

            positives.append(pos)
            negatives.append(neg)
        positives = torch.stack(positives).view(batchSize,1)
        negatives = torch.stack(negatives)
        # print(positives.shape)
        # print(negatives.shape)
        preds = torch.cat((positives, negatives), dim=1) / self.T
        targets = torch.zeros([batchSize]).cuda().long()
        loss = self.criterion(preds, targets)
        return loss


class ViewContrastiveLoss(nn.Module):
    def __init__(self, num_instance=4, T=1.0, mode='one'):
        super(ViewContrastiveLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.num_instance = num_instance
        self.T = T
        self.mode=mode

    def forward(self, q, k, label, k_model_old=None, num_ids_new=None):
        batchSize = q.shape[0]
        if self.mode == 'one':
            rand_idx = self.get_shuffle_ids(batchSize, ranges=self.num_instance)
            # pos logit
            l_pos = torch.bmm(q.view(batchSize, 1, -1), k[rand_idx].view(batchSize, -1, 1))
            l_pos = l_pos.view(batchSize, 1)
            N = q.size(0)
            mat_sim = torch.matmul(q, k.transpose(0, 1))
            # mat_eq = label.expand(N, N).eq(label.expand(N, N).t())
            mat_ne = label.expand(N, N).ne(label.expand(N, N).t())
            # positives = torch.masked_select(mat_sim, mat_eq).view(batchSize, -1)
            negatives = torch.masked_select(mat_sim, mat_ne).view(batchSize, -1)
            out = torch.cat((l_pos, negatives), dim=1)/self.T
            targets = torch.zeros([batchSize]).cuda().long()
            loss = self.criterion(out, targets)
        if self.mode == 'random':
            rand_idx1 = self.get_shuffle_ids(batchSize, ranges=self.num_instance)
            rand_idx2 = self.get_shuffle_ids(batchSize, ranges=self.num_instance)
            rand_idx3 = self.get_shuffle_ids(batchSize, ranges=self.num_instance)
            rand_idx4 = self.get_shuffle_ids(batchSize, ranges=self.num_instance)

            k = (k[rand_idx1]+k[rand_idx2]+k[rand_idx3]+k[rand_idx4])/4

            # pos logit
            l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
            l_pos = l_pos.view(batchSize, 1)
            N = q.size(0)
            mat_sim = torch.matmul(q, k.transpose(0, 1))
            # mat_eq = label.expand(N, N).eq(label.expand(N, N).t())
            mat_ne = label.expand(N, N).ne(label.expand(N, N).t())
            # positives = torch.masked_select(mat_sim, mat_eq).view(batchSize, -1)
            negatives = torch.masked_select(mat_sim, mat_ne).view(batchSize, -1)
            out = torch.cat((l_pos, negatives), dim=1)/self.T
            targets = torch.zeros([batchSize]).cuda().long()
            loss = self.criterion(out, targets)

        elif self.mode == 'hard':
            N = q.size(0)
            mat_sim = torch.matmul(q, k.transpose(0, 1))
            mat_eq = label.expand(N, N).eq(label.expand(N, N).t()).float()
            # batch hard
            hard_p, hard_n, hard_p_indice, hard_n_indice = self.batch_hard(mat_sim, mat_eq, True)
            l_pos = hard_p.view(batchSize, 1)
            l_neg = hard_n.view(batchSize, 1)

            # mat_center = torch.matmul(q, centers.transpose(0, 1))
            # mat_center_ne = label.expand(centers.size(0),N).ne(torch.arange(centers.size(0)).expand(N, centers.size(0)).t().cuda()).t()
            # negatives_center = torch.masked_select(mat_center, mat_center_ne).view(batchSize, -1)

            # mat_outlier = torch.matmul(q, outlier.transpose(0, 1))

            # l = np.random.beta(0.75, 0.75)
            # l = max(l, 1 - l)
            # l_pos = l_pos * l + l_neg * (1-l)
            # l_neg = l_pos * (1-l) + l_neg * l
            mat_ne = label.expand(N, N).ne(label.expand(N, N).t())
            # positives = torch.masked_select(mat_sim, mat_eq).view(-1, 1)
            negatives = torch.masked_select(mat_sim, mat_ne).view(batchSize, -1)
            out = torch.cat((l_pos, negatives), dim=1) / self.T
            # out = torch.cat((l_pos, l_neg, negatives), dim=1) / self.T
            targets = torch.zeros([batchSize]).cuda().long()
            triple_dist = F.log_softmax(out, dim=1)
            triple_dist_ref = torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1), 1)
            # triple_dist_ref = torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1), 1)*l + torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1)+1, 1) * (1-l)
            loss = (- triple_dist_ref * triple_dist).mean(0).sum()
        # elif self.mode == 'lifelong':
        #     N = q.size(0)
        #     mat_sim = torch.matmul(q, k.transpose(0, 1))
        #     mat_eq = label.expand(N, N).eq(label.expand(N, N).t()).float()
        #     # batch hard
        #     hard_p, hard_n, hard_p_indice, hard_n_indice = self.batch_hard(mat_sim, mat_eq, True)
        #     # l_pos = hard_p.view(batchSize, 1)
        #
        #     mat_sim_old = torch.matmul(q, k_model_old.transpose(0, 1))
        #     hard_p_old, hard_n_old, hard_p_indice_old, hard_n_indice_old = self.batch_hard(mat_sim_old, mat_eq, True)
        #     # l_pos_old = hard_p_old.view(batchSize, 1)
        #
        #     l_pos_lifelong = []
        #     for i in range(N):
        #         if label[i] < num_ids_new:
        #             l_pos_lifelong.append(hard_p[i])
        #         else:
        #             l_pos_lifelong.append(hard_p_old[i])
        #     l_pos_lifelong = torch.stack(l_pos_lifelong, dim=0).view(N, 1)
        #
        #     mat_ne = label.expand(N, N).ne(label.expand(N, N).t())
        #     # positives = torch.masked_select(mat_sim, mat_eq).view(-1, 1)
        #     negatives = torch.masked_select(mat_sim, mat_ne).view(batchSize, -1)
        #     negatives_old = torch.masked_select(mat_sim_old, mat_ne).view(batchSize, -1)
        #     out = torch.cat((l_pos_lifelong, negatives, negatives_old), dim=1) / self.T
        #     # out = torch.cat((l_pos, l_neg, negatives), dim=1) / self.T
        #     targets = torch.zeros([batchSize]).cuda().long()
        #     triple_dist = F.log_softmax(out, dim=1)
        #     triple_dist_ref = torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1), 1)
        #     # triple_dist_ref = torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1), 1)*l + torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1)+1, 1) * (1-l)
        #     loss = (- triple_dist_ref * triple_dist).mean(0).sum()
        else:
            N = q.size(0)
            mat_sim = torch.matmul(q, k.transpose(0, 1))
            mat_eq = label.expand(N, N).eq(label.expand(N, N).t())
            mat_ne = label.expand(N, N).ne(label.expand(N, N).t())
            positives = torch.masked_select(mat_sim, mat_eq).view(batchSize, -1)
            negatives = torch.masked_select(mat_sim, mat_ne).view(batchSize, -1)
            positives = torch.topk(positives, k=4, dim=1, largest=True, sorted=True)[0]
            positive = torch.mean(positives[:, 3:4], dim=1, keepdim=True)
            out = torch.cat((positive, negatives), dim=1) / self.T
            targets = torch.zeros([batchSize]).cuda().long()
            loss = self.criterion(out, targets)
            # loss = 0
            # positive_range = [1, 2, 3]
            # for i in positive_range:
            #     out = torch.cat((positives[:,i:i+1], negatives), dim=1) / self.T
            #     targets = torch.zeros([batchSize]).cuda().long()
            #     loss += self.criterion(out, targets)
            # loss = loss/len(positive_range)
        return loss

    def get_shuffle_ids(self, bsz, ranges):
        """sample one random correct idx"""
        rand_inds = torch.zeros(bsz).long().cuda()
        for i in range(bsz//ranges):
            rand_inds[i*ranges:(i+1)*ranges] = i*ranges+torch.randperm(ranges).long().cuda()
        return rand_inds

    def get_negative_ids(self, bsz, ranges):
        """sample one random negative idx"""
        rand_inds = torch.zeros(bsz).long().cuda()
        for i in range(bsz//ranges):
            rand_inds[i*ranges:(i+1)*ranges] = i*ranges+torch.randperm(ranges).long().cuda()
        return rand_inds

    def get_random_ids(self, bsz):
        """sample one random idx"""
        rand_inds = torch.randperm(bsz).long().cuda()
        return rand_inds

    def batch_hard(self, mat_sim, mat_eq, indice=False):
        sorted_mat_sim, positive_indices = torch.sort(mat_sim + (9999999.) * (1 - mat_eq), dim=1,
                                                           descending=False)
        hard_p = sorted_mat_sim[:, 0]
        hard_p_indice = positive_indices[:, 0]
        sorted_mat_distance, negative_indices = torch.sort(mat_sim + (-9999999.) * (mat_eq), dim=1,
                                                           descending=True)
        hard_n = sorted_mat_distance[:, 0]
        hard_n_indice = negative_indices[:, 0]
        if (indice):
            return hard_p, hard_n, hard_p_indice, hard_n_indice
        return hard_p, hard_n

