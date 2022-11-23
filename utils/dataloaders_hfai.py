# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import pickle
import yaml
import os
import random
from pathlib import Path
import copy

import numpy as np
import torch
from torch.utils.data import distributed

from pycocotools.coco import COCO
from ffrecord import FileReader
from ffrecord.torch import Dataset, DataLoader

from utils.augmentations import (Albumentations, augment_hsv, letterbox)
from utils.general import (cv2, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders


def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(split,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=True,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False):

    assert split in ['train', 'val']

    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(
            split,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            image_weights=image_weights,
            check_data=True)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle and sampler is None,
                      num_workers=nw,
                      sampler=sampler,
                      pin_memory=PIN_MEMORY,
                      collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
                      worker_init_fn=seed_worker,
                      generator=generator), dataset


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):
    # YOLOv5_hfai train_loader/val_loader, loads images and labels for training and validation

    def __init__(self,
                 split,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 check_data: bool = True):

        self.data_dir = Path("/public_dataset/1/ffdataset/COCO/")

        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.single_cls = single_cls
        self.stride = stride
        self.split = split
        self.albumentations = Albumentations(size=img_size) if augment else None

        self.reader = FileReader(self.data_dir / f"{split}2017.ffr", check_data)
        self.coco = COCO(self.data_dir / f"annotations/instances_{self.split}2017.json")
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.indices = range(self.__len__())
        self.n = self.__len__()


    def load_images(self, indices):
        bytes_ = self.reader.read(indices)
        imgs = []
        for x in bytes_:
            im = pickle.loads(x).convert("RGB")
            im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            im = im.transpose((1, 0, 2))
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
            imgs.append((im, (h0, w0), im.shape[:2]))   # im, hw_original, hw_resized
        return imgs


    def load_annos(self, indices):
        labels = []
        for index in indices:
            img_id = self.ids[index]
            ann_id = self.coco.getAnnIds(img_id)
            ann = self.coco.loadAnns(ann_id)
            label = []
            for item in ann:
                label.append([item['category_id']] + item['bbox'])
            labels.append(label)
        return labels, []


    def __len__(self):
        return self.reader.n


    def __getitem__(self, indices):

        imgs, labels, shapes = [], [], []

        hyp = self.hyp
        # Load image
        tmp_imgs = self.load_images(indices)
        tmp_labels, tmp_segments = self.load_annos(indices)
        for i, (img, (h0, w0), (h, w)) in enumerate(tmp_imgs):
            shape = self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shape = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
            label = np.asarray(copy.deepcopy(tmp_labels[i]))
            imgs.append(img)
            labels.append(label)
            shapes.append(shape)

        for i, (img, label) in enumerate(zip(imgs, labels)):
            nl = len(label)  # number of labels
            if nl:
                label[:, 1:5] = xyxy2xywhn(label[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)
            if self.augment:
                # Albumentations
                img, label = self.albumentations(img, label)
                nl = len(label)  # update after albumentations
                # HSV color-space
                augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
                # Flip up-down
                if random.random() < hyp['flipud']:
                    img = np.flipud(img)
                    if nl:
                        label[:, 2] = 1 - label[:, 2]
                # Flip left-right
                if random.random() < hyp['fliplr']:
                    img = np.fliplr(img)
                    if nl:
                        label[:, 1] = 1 - label[:, 1]
            label_out = torch.zeros((nl, 6))
            if nl:
                label_out[:, 1:] = torch.from_numpy(label)
            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            imgs[i] = img
            labels[i] = label_out

        return imgs, labels, shapes


    @staticmethod
    def collate_fn(batch):
        imgs, labels, shapes = batch  # transposed
        for i, lb in enumerate(labels):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.from_numpy(np.stack(imgs, 0)), torch.cat(labels, 0), shapes


if __name__ == '__main__':
    with open('../data/hyps/hyp.scratch-low.yaml', errors='ignore') as f:
        hyp = yaml.safe_load(f)
    # dataset = LoadImagesAndLabels('train', hyp=hyp)
    # items = dataset[[0, 1, 2]]
    # print(items)

    loader, dataset = create_dataloader('train', 640, 16, 32, hyp=hyp)
    batch_imgs, batch_labels, batch_shapes = next(iter(loader))
    print(batch_imgs.size(), batch_labels.size())

