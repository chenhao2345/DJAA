from __future__ import print_function, absolute_import

import os
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
import copy

from sklearn.cluster import DBSCAN
import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from djaa.utils.logging import Logger
from djaa import datasets
from djaa import models
from djaa.trainers import ImageTrainer_memory
from djaa.evaluators import Evaluator, extract_features
from djaa.utils.data import IterLoader
from djaa.utils.data import transforms as T
from djaa.utils.data.sampler import MoreCameraSampler, RandomIdentitySampler
from djaa.utils.data.preprocessor import Preprocessor_index, Preprocessor
from djaa.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from djaa.utils.faiss_rerank import compute_jaccard_distance
from djaa.utils.lr_scheduler import WarmupMultiStepLR
from sklearn.metrics import normalized_mutual_info_score
from operator import itemgetter, attrgetter

start_epoch = best_mAP = 0


def get_data(name, data_dir, split=None):
    kwargs = {"split": split}
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root, **kwargs)
    return dataset


def get_train_loader(dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, mutual=False, index=False):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.GaussianBlur([.1, 2.])], p=0.5),
        # T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
    ])

    weak_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalizer,
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        # sampler = RandomIdentitySampler(train_set, num_instances, video=True)
        sampler = MoreCameraSampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor_index(train_set, root=dataset.images_dir, transform=train_transformer, mutual=mutual,
                                      index=index, transform2=weak_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model_1 = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=0,
                            pooling_type=args.pooling_type)

    model_1_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=0,
                                pooling_type=args.pooling_type)

    model_1.cuda()
    model_1_ema.cuda()
    model_1 = nn.DataParallel(model_1)
    model_1_ema = nn.DataParallel(model_1_ema)

    if args.init != '':
        initial_weights = load_checkpoint(args.init)
        copy_state_dict(initial_weights['state_dict'], model_1)
        copy_state_dict(initial_weights['state_dict'], model_1_ema)

    return model_1, model_1_ema


def lifelong_unsupervised_trainer(args, dataset_target, model_1, model_1_ema, optimizer, lr_scheduler, evaluator_1_ema,
                                  model_1_old=None, centers_old=None, dictionary_old=None):
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, 256, args.workers)
    labels_lastepoch = []

    for epoch in range(args.epochs):

        cluster_loader = get_test_loader(dataset_target, args.height, args.width, 256, args.workers,
                                         testset=dataset_target.train)
        dict_f1, _ = extract_features(model_1_ema, cluster_loader, print_freq=50)
        cf = torch.stack(list(dict_f1.values()))
        rerank_dist = compute_jaccard_distance(cf, k1=args.k1, k2=6)
        eps = args.rho
        print('eps in cluster: {:.3f}'.format(eps))
        print('Clustering and labeling...')
        cluster = DBSCAN(eps=eps, min_samples=args.min_samples, metric='precomputed', n_jobs=-1)
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - (1 if -1 in labels else 0)

        centers = []
        for id in range(num_ids):
            center = torch.mean(cf[labels == id], dim=0)
            centers.append(center)
        centers = torch.stack(centers, dim=0)

        # change pseudo labels
        pseudo_labeled_dataset = []
        pseudo_outliers = 0
        labels_true = []
        cams = []
        # id_cam_centers = {}
        for i, ((fname, pid, cid), label, feat) in enumerate(zip(dataset_target.train, labels, cf)):
            labels_true.append(pid)
            cams.append(cid)
            if label == -1:
                pseudo_outliers += 1
            else:
                pseudo_labeled_dataset.append((fname, label.item(), cid, feat))
                # print(feat)
        cams = np.asarray(cams)
        num_cams = len(np.unique(cams))

        id_cam_centers = {}
        for id in range(num_ids):
            id_cam_centers[id] = []
            for cam in np.unique(cams):
                mask = np.logical_and(labels == id, cams == cam)
                if any(mask):
                    id_cam_centers[id].append(torch.mean(cf[mask], dim=0))

        pseudo_labeled_dataset_newold = []
        pseudo_labeled_dataset_newold.extend(pseudo_labeled_dataset)
        pseudo_labeled_dataset_old = []
        num_ids_newold = num_ids
        num_img_old = 0
        if (centers_old is not None) and (dictionary_old is not None):
            num_ids_newold += centers_old.size(0)
            centers = torch.cat([centers, centers_old], dim=0)
            old_pid = []
            for j, per_id in enumerate(dictionary_old.values()):
                for (fname, pid, cid, sim) in per_id:
                    num_img_old += 1
                    old_pid.append(pid + num_ids)
                    pseudo_labeled_dataset_old.append((fname, pid + num_ids, cid, sim))
                    pseudo_labeled_dataset_newold.append((fname, pid + num_ids, cid, sim))
            print('Epoch {}, old dataset has {} labeled samples of {} ids'.
                  format(epoch, num_img_old, len(set(old_pid))))

        # del rerank_dist, dict_f1, cf

        print('Label score:', normalized_mutual_info_score(labels_true=labels_true, labels_pred=labels))
        if epoch > 0:
            print('Label score current/last epoch:',
                  normalized_mutual_info_score(labels_true=labels, labels_pred=labels_lastepoch[-1]))
        labels_lastepoch.append(labels)
        print('Epoch {}, current dataset has {} labeled samples of {} ids and {} unlabeled samples'.
              format(epoch, len(pseudo_labeled_dataset), num_ids, pseudo_outliers))
        print('Totally, epoch {} has {} labeled samples of {} ids'.
              format(epoch, len(pseudo_labeled_dataset_newold), num_ids_newold))
        print('Learning Rate:', optimizer.param_groups[0]['lr'])
        train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                               args.batch_size, args.workers, args.num_instances, args.iters,
                                               trainset=pseudo_labeled_dataset, mutual=True, index=True)
        train_loader_target.new_epoch()

        if (centers_old is not None) and (dictionary_old is not None):
            train_loader_target_old = get_train_loader(dataset_target, args.height, args.width,
                                                       args.batch_size, args.workers, args.mem_samples, args.iters,
                                                       trainset=pseudo_labeled_dataset_old, mutual=True, index=True)
            train_loader_target_old.new_epoch()
            train_loader_target_newold = get_train_loader(dataset_target, args.height, args.width,
                                                          args.batch_size, args.workers, args.mem_samples, args.iters,
                                                          trainset=pseudo_labeled_dataset_newold, mutual=True,
                                                          index=True)
            train_loader_target_newold.new_epoch()
        else:
            train_loader_target_old = None
            train_loader_target_newold = None

        # Trainer
        trainer = ImageTrainer_memory(model_1, model_1_ema, num_cluster=num_ids_newold, alpha=args.alpha,
                                      num_instance=args.num_instances, tau_c=args.tau_c, tau_v=args.tau_v,
                                      memory=None, scale_kl=args.scale_kl, model_1_old=model_1_old,
                                      lambda_kl=args.lambda_kl)

        trainer.train(epoch, train_loader_target, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_target), centers=centers,
                      id_cam_centers=id_cam_centers, centers_model_old=None, num_ids_new=num_ids,
                      train_loader_target_old=train_loader_target_old,
                      train_loader_target_newold=train_loader_target_newold)

        lr_scheduler.step()
        if (epoch + 1) % args.eval_step == 0:
            cmc, mAP_1 = evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
                                                  cmc_flag=True)

    pseudo_labeled_dictionary = {}
    new_dictionary = {}
    # update_num=0.3*512
    # update_num=0.1*num_ids
    pid2num = {}

    for i, (fname, label, cid, feat) in enumerate(pseudo_labeled_dataset):
        sim = F.cosine_similarity(feat.view(1, -1).cuda(), centers[label].view(1, -1).cuda()).item()
        # if label in randperm:
        if label not in pseudo_labeled_dictionary:
            pseudo_labeled_dictionary[label] = list()
        pseudo_labeled_dictionary[label].append((fname, label, cid, feat, sim))

    for j in pseudo_labeled_dictionary.keys():
        pid2num[j] = len(pseudo_labeled_dictionary[j])
        ## max similarity
        pseudo_labeled_dictionary[j] = [(fname, label, cid, feat) for (fname, label, cid, feat, sim) in
                                        sorted(pseudo_labeled_dictionary[j], key=itemgetter(4), reverse=True)[
                                        :args.mem_samples]]

        # ## min similarity
        # pseudo_labeled_dictionary[j] = [(fname, label, cid, feat) for (fname, label, cid, feat, sim) in
        #                                 sorted(pseudo_labeled_dictionary[j], key=itemgetter(4), reverse=False)[
        #                                 :args.mem_samples]]

        # # random
        # # if len(pseudo_labeled_dictionary[j]) >= args.mem_samples:
        # pseudo_labeled_dictionary[j] = [(fname, label, cid, feat) for (fname, label, cid, feat, sim) in
        #                                 random.choices(pseudo_labeled_dictionary[j], k=args.mem_samples)]

    if (centers_old is not None) and (dictionary_old is not None):
        for key, val in dictionary_old.items():
            temp_list = []
            for (fname, pid, cid, feat) in val:
                temp_list.append((fname, pid + num_ids, cid + num_cams, feat))
            dictionary_old[key] = temp_list

        for k in dictionary_old.keys():
            pseudo_labeled_dictionary[k + num_ids] = dictionary_old[k]

    ## reduce new + old data to store
    mem_num = args.mem_num
    # update_num = mem_num//2
    update_num = mem_num * (num_ids) / (mem_num + num_ids)
    if len(pseudo_labeled_dictionary) > mem_num:
        # ## select centers by sim between dataset center and id centers
        # sim_id = F.cosine_similarity(dataset_center.cuda(), centers_new.cuda())
        # val,idx = torch.topk(sim_id, k=update_num, largest=False, sorted=True)
        # idx = idx.tolist()
        # if len(idx)<update_num:
        #     old_key = np.random.permutation(np.arange(num_ids,len(pseudo_labeled_dictionary)))[:update_num-len(idx)]
        #     idx.extend(old_key.tolist())

        ## select centers by cluster instance number
        if (centers_old is not None) and (dictionary_old is not None):
            val, idx = torch.topk(torch.tensor(list(pid2num.values())), k=int(update_num), largest=True, sorted=True)
            new_key = idx.tolist()
            old_key = np.random.permutation(np.arange(num_ids, len(pseudo_labeled_dictionary)))[
                          :mem_num - int(update_num)]
            perm = np.concatenate((new_key, old_key))
        else:
            val, idx = torch.topk(torch.tensor(list(pid2num.values())), k=mem_num, largest=True, sorted=True)
            perm = idx.tolist()
        idx = perm[:mem_num]

        # ## random centers
        # if (centers_old is not None) and (dictionary_old is not None):
        #     # perm = np.random.permutation(list(pseudo_labeled_dictionary.keys()))
        #     new_key = np.random.permutation(np.arange(num_ids))[:int(update_num)]
        #     old_key = np.random.permutation(np.arange(num_ids, len(pseudo_labeled_dictionary)))[
        #               :mem_num - int(update_num)]
        #     perm = np.concatenate((new_key, old_key))
        # else:
        #     perm = np.random.permutation(list(pseudo_labeled_dictionary.keys()))
        # idx = perm[:mem_num]

        for id, num in enumerate(idx):
            # new_dictionary[id] = pseudo_labeled_dictionary[num]
            temp_list = []
            for (fname, pid, cid, feat) in pseudo_labeled_dictionary[num]:
                temp_list.append((fname, id, cid, feat))
            new_dictionary[id] = temp_list
        centers = centers[idx]

        market_count, cuhk_count, msmt_count = 0, 0, 0
        for j, per_id in enumerate(new_dictionary.values()):
            for (fname, pid, cid, _) in per_id:
                if 'msmt17' in fname:
                    msmt_count += 1
                elif 'market1501' in fname:
                    market_count += 1
                elif 'cuhk-sysu' in fname:
                    cuhk_count += 1
        print('Memory Update')
        print('market_count:{}, cuhk_count:{}, msmt_count:{}'.format(market_count, cuhk_count, msmt_count))
        return model_1, model_1_ema, centers, new_dictionary

    market_count, cuhk_count, msmt_count = 0, 0, 0
    for j, per_id in enumerate(pseudo_labeled_dictionary.values()):
        for (fname, pid, cid, _) in per_id:
            if 'msmt17' in fname:
                msmt_count += 1
            elif 'market1501' in fname:
                market_count += 1
            elif 'cuhk-sysu' in fname:
                cuhk_count += 1
    print('Memory Update')
    print('market_count:{}, cuhk_count:{}, msmt_count:{}'.format(market_count, cuhk_count, msmt_count))
    return model_1, model_1_ema, centers, pseudo_labeled_dictionary

def evaluate_all(args, datasets, evaluator_1_ema):
    rank1 = []
    rank5 = []
    rank10 = []
    mAP = []
    for dataset in datasets:
        print('Test on', dataset.dataset_dir)
        test_loader = get_test_loader(dataset, args.height, args.width, 256, args.workers)
        cmc, mAP_1 = evaluator_1_ema.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
        rank1.append(cmc[0])
        rank5.append(cmc[4])
        rank10.append(cmc[9])
        mAP.append(mAP_1)
    print('Average:')
    print('mAP:', sum(mAP) / len(mAP) * 100)
    print('rank1:', sum(rank1) / len(rank1) * 100)
    print('rank5:', sum(rank5) / len(rank5) * 100)
    print('rank10:', sum(rank10) / len(rank10) * 100)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    dataset_market = get_data('market1501', args.data_dir, split=0)
    # dataset_duke = get_data('dukemtmc-reid', args.data_dir)
    dataset_cuhksysu = get_data('cuhk-sysu', args.data_dir, split=0)
    dataset_msmt = get_data('msmt17', args.data_dir, split=0)

    # dataset_market2 = get_data('market1501', args.data_dir, split=2)
    # # dataset_duke = get_data('dukemtmc-reid', args.data_dir)
    # dataset_cuhksysu2 = get_data('cuhk-sysu', args.data_dir, split=2)
    # dataset_msmt2 = get_data('msmt17', args.data_dir, split=2)

    dataset_ilids = get_data('ilids', args.data_dir)
    dataset_viper = get_data('viper', args.data_dir)
    dataset_prid2011 = get_data('prid2011', args.data_dir)
    dataset_grid = get_data('grid', args.data_dir)
    dataset_cuhk01 = get_data('cuhk01', args.data_dir)
    dataset_cuhk02 = get_data('cuhk02', args.data_dir)
    dataset_sensereid = get_data('sensereid', args.data_dir)
    dataset_cuhk03 = get_data('cuhk03', args.data_dir)
    dataset_3dpes = get_data('3dpes', args.data_dir)

    # dataset_mmp_cafe = get_data('mmptracking_cafe', args.data_dir)
    # dataset_mmp_industry = get_data('mmptracking_industry', args.data_dir)
    # dataset_mmp_lobby = get_data('mmptracking_lobby', args.data_dir)
    # dataset_mmp_office = get_data('mmptracking_office', args.data_dir)
    # dataset_mmp_retail = get_data('mmptracking_retail', args.data_dir)

    datasets = [dataset_market, dataset_cuhksysu, dataset_msmt]
    # datasets = [dataset_msmt, dataset_market, dataset_cuhksysu] # order2
    # datasets_unseen = [dataset_cuhk03, dataset_ilids, dataset_viper, dataset_3dpes]
    datasets_eva = [dataset_market, dataset_cuhksysu, dataset_msmt]
    datasets_unseen = [dataset_viper, dataset_prid2011, dataset_grid, dataset_ilids, dataset_cuhk01, dataset_cuhk02,
                       dataset_sensereid, dataset_cuhk03, dataset_3dpes]

    # datasets_unseen_mmp = [dataset_mmp_cafe, dataset_mmp_industry, dataset_mmp_lobby, dataset_mmp_office, dataset_mmp_retail]
    # Create model
    model_1, model_1_ema = create_model(args)

    # Optimizer
    params = []
    for key, value in model_1.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)

    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=1, warmup_factor=0.1,
                                     warmup_iters=args.warmup_step)

    # Evaluator
    evaluator_1_ema = Evaluator(model_1_ema)
    # evaluate_all(args, datasets_unseen, evaluator_1_ema)

    model_1, model_1_ema, centers, pseudo_labeled_dictionary = lifelong_unsupervised_trainer(
        args,
        datasets[0],
        model_1,
        model_1_ema,
        optimizer,
        lr_scheduler,
        evaluator_1_ema)
    evaluate_all(args, datasets_eva, evaluator_1_ema)
    evaluate_all(args, datasets_unseen, evaluator_1_ema)
    # evaluate_all(args, datasets_unseen_mmp, evaluator_1_ema)
    save_dir = osp.join(args.logs_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_checkpoint({'state_dict': model_1_ema.state_dict()}, False, fpath=osp.join(save_dir, 'step{}.pth.tar'.format(1)))
    model_1_old = copy.deepcopy(model_1_ema)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=1, warmup_factor=0.1,
                                     warmup_iters=args.warmup_step)

    for (k_q, v_q), (k_k, v_k) in zip(model_1_ema.state_dict().items(), model_1.state_dict().items()):
        assert k_k == k_q, "state_dict names are different!"
        v_k.copy_(v_q)

    for i, dataset in enumerate(datasets[1:]):
        print('Training on dataset # {}.'.format(i + 2))
        model_1, model_1_ema, centers, pseudo_labeled_dictionary = lifelong_unsupervised_trainer(
            args, dataset, model_1, model_1_ema, optimizer, lr_scheduler, evaluator_1_ema, model_1_old=model_1_old,
            centers_old=centers, dictionary_old=pseudo_labeled_dictionary)
        evaluate_all(args, datasets_eva, evaluator_1_ema)
        evaluate_all(args, datasets_unseen, evaluator_1_ema)
        # evaluate_all(args, datasets_unseen_mmp, evaluator_1_ema)
        save_checkpoint({'state_dict': model_1_ema.state_dict()}, False, fpath=osp.join(save_dir, 'step{}.pth.tar'.format(i+2)))

        model_1_old = copy.deepcopy(model_1_ema)
        lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=1, warmup_factor=0.1,
                                         warmup_iters=args.warmup_step)

        for (k_q, v_q), (k_k, v_k) in zip(model_1_ema.state_dict().items(), model_1.state_dict().items()):
            assert k_k == k_q, "state_dict names are different!"
            v_k.copy_(v_q)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Moco Training")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-ds', '--dataset-source', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--pooling-type', type=str, default='gem')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters")
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=400)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--tau-c', type=float, default=0.5)
    parser.add_argument('--tau-v', type=float, default=0.09)
    parser.add_argument('--scale-kl', type=float, default=2.0)
    parser.add_argument('--lambda-kl', type=float, default=20.0)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[],
                        help='milestones for the learning rate decay')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    # cluster
    parser.add_argument('--rho', type=float, default=0.55,
                        help="rho percentage")
    parser.add_argument('--k1', type=int, default=30,
                        help="k1, default: 30")
    parser.add_argument('--min-samples', type=int, default=4,
                        help="min sample, default: 4")
    parser.add_argument('--mem-samples', type=int, default=1,
                        help="mem samples per person, default: 4")
    parser.add_argument('--mem-num', type=int, default=512,
                        help="mem person number, default: 512")
    # init
    parser.add_argument('--init', type=str,
                        default='',
                        metavar='PATH')
    end = time.time()
    main()
    print('Time used: {}'.format(time.time() - end))
