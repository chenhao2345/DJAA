from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
from scipy.io import loadmat
from ..utils.data import BaseImageDataset
import numpy as np
import torch
from PIL import Image
from collections import OrderedDict, defaultdict
import copy
import os
from tqdm import tqdm
import numpy as np

class CUHKSYSU(BaseImageDataset):
    """
        cuhk-sysu for person search
    """

    def __init__(self, root='', verbose=True, use_subset_train=False, **kwargs):
        super(CUHKSYSU, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, 'CUHK-SYSU')
        self.data_path = osp.join(self.dataset_dir, 'Image', 'SSM')
        self.annotation_path = osp.join(self.dataset_dir, 'annotation')
        required_files = [
            self.root, self.dataset_dir, self.data_path, self.annotation_path
        ]
        self.split = kwargs['split']
        self.check_before_run(required_files)

        self.processed_dir = osp.join(self.dataset_dir, 'cuhksysu4reid')
        self.processed_dir_train = osp.join(self.processed_dir, 'train')
        self.processed_dir_query = osp.join(self.processed_dir, 'query')
        self.processed_dir_gallery = osp.join(self.processed_dir, 'gallery')
        self.processed_dir_combine = osp.join(self.processed_dir, 'combine')
        required_files_state = [self.processed_dir_train, self.processed_dir_query, self.processed_dir_gallery, self.processed_dir_combine]

        if all(map(osp.exists, required_files_state)):
            self.train = self.process_dir(self.processed_dir_train, relabel=True)
            self._combine = self.process_dir(self.processed_dir_combine, relabel=True)
            self.query = self.process_query_dir(self.processed_dir_query, relabel=False)
            self.gallery = self.process_dir(self.processed_dir_gallery, relabel=False)
        else:
            if osp.exists(self.processed_dir) is False:
                os.mkdir(self.processed_dir)
            os.mkdir(self.processed_dir_train)
            os.mkdir(self.processed_dir_combine)
            os.mkdir(self.processed_dir_query)
            os.mkdir(self.processed_dir_gallery)

            self.preprocessing()
            self.train = self.process_dir(self.processed_dir_train, relabel=True)
            self._combine = self.process_dir(self.processed_dir_combine, relabel=True)
            self.query = self.process_query_dir(self.processed_dir_query, relabel=False)
            self.gallery = self.process_dir(self.processed_dir_gallery, relabel=False)
        if use_subset_train:
            self.sub_set()
        if verbose:
            print("=> Cuhk-sysu loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


            # print('For conbine, cuhksysu_gallery == query')

    def _relabels_incremental(self, samples, label_index, is_mix=False):
        '''
        reorder labels
        map labels [1, 3, 5, 7] to [0,1,2,3]
        '''
        ids = []
        pid2label = {}
        for sample in samples:
            ids.append(sample[label_index])
        # delete repetitive elments and order
        ids = list(set(ids))
        ids.sort()

        # reorder
        for sample in samples:
            sample = list(sample)
            pid2label[sample[label_index]] = ids.index(sample[label_index])
        new_samples = copy.deepcopy(samples)
        for i, sample in enumerate(samples):
            new_samples[i] = list(new_samples[i])
            new_samples[i][label_index] = pid2label[sample[label_index]]
        if is_mix:
            return samples, pid2label
        else:
            return new_samples

    def sub_set(self):
        results, bigger4_list, sub_train = {}, [], []
        for it in self.train:
            if it[1] not in results.keys():
                results[it[1]] = 1
            else:
                results[it[1]] += 1
        for key, value in results.items():
            if value >= 4:
                bigger4_list.append(key)
        if self.split == 1:
            selected = bigger4_list[:350]
        if self.split == 2:
            selected = bigger4_list[350:700]
        if self.split == 0:
            selected = bigger4_list
        for it in self.train:
            if it[1] in bigger4_list and it[1] in selected:
                sub_train.append(it)
        sub_train = self._relabels_incremental(sub_train, 1, is_mix=False)
        self.train = sub_train

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_s([-\d]+)_([-\d]+)_([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, image_name, bbox_index, is_hard = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        # selected = np.random.permutation(len(pid2label))[:350]
        for img_path in img_paths:
            pid, _, _, _ = map(int, pattern.search(img_path).groups())
            if relabel:
                pid = pid2label[pid]
            # if relabel and pid not in selected:
            #     continue
            data.append((img_path, pid, 0))

        return data

    def process_query_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_s([-\d]+)_([-\d]+)_([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, image_name, bbox_index, is_hard = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, _, _, _ = map(int, pattern.search(img_path).groups())
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, 1))

        return data

    def crop_store(self, data_dict, save_dir):
        def _crop_store():
            name = save_dir.split('/')[-1]
            image_dict = defaultdict(list)

            index_instance = 0
            for key, person_images in tqdm(data_dict.items()):
                for image_path, box, pid_name, pid, im_name, is_hard in person_images:
                    assert osp.exists(image_path)
                    one_img = Image.open(image_path)
                    one_img_copy = copy.deepcopy(one_img)
                    box_tuple = tuple(box.round())
                    box_tuple = map(int, box_tuple)
                    filled_pid = str(pid).zfill(5)
                    is_hard = str(is_hard)
                    cropped = one_img_copy.crop(box_tuple)
                    image_name = im_name.replace('.jpg', '')
                    cropped_path = osp.join(save_dir,
                                            f'{filled_pid}_{image_name}_{str(index_instance).zfill(7)}_{is_hard}.jpg')
                    cropped.save(cropped_path)
                    image_dict[pid_name].append((cropped_path, int(pid), 0, 'cuhksysu', int(pid)))
                    index_instance = index_instance + 1

            print(f'Finished processing {name} dir!')
            return image_dict

        if osp.exists(save_dir) is False:
            os.makedirs(save_dir)
            _crop_store()
        else:
            _crop_store()


    def preprocessing(self):
        Train_mat = loadmat(osp.join(self.annotation_path, 'test', 'train_test', 'Train.mat'))
        # testg50_mat = loadmat(osp.join(self.annotation_path, 'test', 'train_test', 'TestG50.mat'))['TestG50'].squeeze()
        testg50_mat = loadmat(osp.join(self.annotation_path, 'test', 'train_test', 'TestG100.mat'))['TestG100'].squeeze()
        all_imgs_mat = loadmat(osp.join(self.annotation_path, 'Person.mat'))
        # pool_mat = loadmat(osp.join(self.annotation_path, 'pool.mat'))
        id_name_to_pid = {}
        train_pid_dict = defaultdict(list)

        train = Train_mat['Train'].squeeze()
        n_train = 0
        for index, item in enumerate(train):
            pid_name = item[0, 0][0][0]
            pid = int(pid_name[1:])
            id_name_to_pid[pid_name] = pid
            scenes = item[0, 0][2].squeeze()
            for im_name, box, is_hard in scenes:
                im_name = str(im_name[0])
                is_hard = is_hard[0][0]
                box = box.squeeze().astype(np.int32)
                box[2:] += box[:2]
                image_path = osp.join(self.data_path, im_name)
                train_pid_dict[pid_name].append((image_path, box, pid_name, pid, im_name, is_hard))
                n_train = n_train + 1

        probe_pid_dict = defaultdict(list)
        gallery_pid_dict = defaultdict(list)
        n_probe = 0
        n_gallery = 0
        for query, gallery in zip(testg50_mat['Query'], testg50_mat['Gallery']):
            im_name = str(query['imname'][0, 0][0])
            roi = query['idlocate'][0, 0][0].astype(np.int32)
            roi[2:] += roi[:2]
            is_hard = query['ishard'][0, 0][0, 0]
            pid_name = query['idname'][0, 0][0]
            pid = int(pid_name[1:])
            assert pid_name not in id_name_to_pid.keys()
            id_name_to_pid[pid_name] = pid
            # im_name, bbox, is_hard, idname, flipped
            image_path = osp.join(self.data_path, im_name)
            probe_pid_dict[pid_name].append((image_path, roi, pid_name, pid, im_name, is_hard))
            n_probe = n_probe + 1
            gallery = gallery.squeeze()
            for _gallery in gallery:
                _im_name = str(_gallery['imname'][0])
                _roi = _gallery['idlocate'][0].astype(np.int32)
                if _roi.size == 0:
                    continue
                else:
                    _roi[2:] += _roi[:2]
                    _is_hard = _gallery['ishard'][0][0]
                    # _id_name = _gallery['idname'][0]
                    # im_name, bbox, is_hard, idname, flipped
                    _image_path = osp.join(self.data_path, _im_name)
                    gallery_pid_dict[pid_name].append((_image_path, _roi, pid_name, pid, _im_name, _is_hard))
                    n_gallery = n_gallery + 1

        num_total_pid = len(train_pid_dict) + len(probe_pid_dict)

        print(num_total_pid)
        all_image_dict = defaultdict(list)
        all_imgs = all_imgs_mat['Person'].squeeze()

        n = 0
        for id_name, _, scenes in all_imgs:
            pid_name = id_name[0]
            pid = int(pid_name[1:])
            scenes = scenes.squeeze()
            for im_name, box, is_hard in scenes:
                im_name = str(im_name[0])
                is_hard = is_hard[0, 0]
                box = box.squeeze().astype(np.int32)
                box[2:] += box[:2]
                image_path = osp.join(self.data_path, im_name)
                all_image_dict[pid_name].append((image_path, box, pid_name, pid, im_name, is_hard))
                n = n + 1

        print(n)
        print(f'n_train: {n_train}, n_probe: {n_probe}, n_gallery: {n_gallery} n_all:{n}')
        train_dict = self.crop_store(train_pid_dict, osp.join(self.processed_dir, 'train'))
        probe_dict = self.crop_store(probe_pid_dict, osp.join(self.processed_dir, 'query'))
        gallery_dict = self.crop_store(gallery_pid_dict, osp.join(self.processed_dir, 'gallery'))
        all_dict = self.crop_store(all_image_dict, osp.join(self.processed_dir, 'combine'))
        