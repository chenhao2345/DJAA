from __future__ import print_function, absolute_import
import os.path as osp
import glob
import random
import re
import shutil
import os
from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json
import numpy as np

class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = ''

    def __init__(self, root, verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.split = kwargs['split']
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        # selected = np.random.permutation(len(pid2label))[:350]
        if self.split == 1:
            selected = np.arange(350)
        if self.split == 2:
            selected = np.arange(350, 700)
        if self.split == 0:
            selected = np.random.permutation(len(pid2label))
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            if relabel and pid not in selected:
                continue
            if relabel and self.split == 2:
                pid -= 350
            dataset.append((img_path, pid, camid))
        return dataset


# class Market1501(BaseImageDataset):
#     """
#     Market1501
#     Reference:
#     Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
#     URL: http://www.liangzheng.org/Project/project_reid.html
#
#     Dataset statistics:
#     # identities: 1501 (+1 for background)
#     # images: 12936 (train) + 3368 (query) + 15913 (gallery)
#     """
#     dataset_dir = ''
#
#     def __init__(self, root, verbose=True, **kwargs):
#         super(Market1501, self).__init__()
#         self.dataset_dir = osp.join(root, self.dataset_dir)
#         self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
#         self.query_dir = osp.join(self.dataset_dir, 'query')
#         self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
#         self.split = kwargs['split']
#         self._check_before_run()
#
#         train = self._process_dir(self.train_dir, relabel=True)
#         query = self._process_dir(self.query_dir, relabel=False)
#         gallery = self._process_dir(self.gallery_dir, relabel=False)
#
#         if verbose:
#             print("=> Market1501 loaded")
#             self.print_dataset_statistics(train, query, gallery)
#
#         if self.split == 1:
#             self.train = train1
#         else:
#             self.train = train2
#         # self.train = train
#         self.query = query
#         self.gallery = gallery
#
#         print(self.train)
#         input()
#
#         self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
#         self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
#         self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
#
#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not osp.exists(self.dataset_dir):
#             raise RuntimeError("'{}' is not available".format(self.dataset_dir))
#         if not osp.exists(self.train_dir):
#             raise RuntimeError("'{}' is not available".format(self.train_dir))
#         if not osp.exists(self.query_dir):
#             raise RuntimeError("'{}' is not available".format(self.query_dir))
#         if not osp.exists(self.gallery_dir):
#             raise RuntimeError("'{}' is not available".format(self.gallery_dir))
#
#     def _process_dir(self, dir_path, relabel=False):
#         img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
#         pattern = re.compile(r'([-\d]+)_c(\d)')
#
#         pid_container = set()
#         for img_path in img_paths:
#             pid, _ = map(int, pattern.search(img_path).groups())
#             if pid == -1: continue  # junk images are just ignored
#             pid_container.add(pid)
#
#         if self.split:
#             pid_container_list = list(pid_container)
#             print(pid_container_list)
#             random.shuffle(pid_container_list)
#             selected1 = set(pid_container_list[:350])
#             selected2 = set(pid_container_list[350:700])
#             pid2label1 = {pid: label for label, pid in enumerate(selected1)}
#             pid2label2 = {pid: label for label, pid in enumerate(selected2)}
#             print(pid2label1)
#             print(pid2label2)
#
#             input()
#
#         train_dataset1 = []
#         train_dataset2 = []
#         for img_path in img_paths:
#             pid, camid = map(int, pattern.search(img_path).groups())
#             if pid == -1: continue  # junk images are just ignored
#             assert 0 <= pid <= 1501  # pid == 0 means background
#             assert 1 <= camid <= 6
#             camid -= 1  # index starts from 0
#             if relabel:
#                 pid = pid2label[pid]
#             if relabel and pid not in selected:
#                 continue
#
#             dataset.append((img_path, pid, camid))
#         return dataset