"""
Dataset for Training and Test
Extended from ADNet code by Hansen et al.
"""
from torch.utils.data import Dataset
import PIL.Image as Image
from torchvision import transforms
import glob
import os
import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F
from .dataset_specifics import *


class DatasetCHAO(Dataset):

    def __init__(self, args):

        # reading the paths
        if args['dataset'] == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'CHAOST2/chaos_MR_T2_normalized/image*'))

        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))

        # remove test fold!
        self.FOLD = get_folds(args['dataset'])
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx in self.FOLD[args['eval_fold']]]

        # split into support/query
        idx = np.arange(len(self.image_dirs))
        self.support_dir = self.image_dirs[idx[args['supp_idx']]]
        self.image_dirs.pop(idx[args['supp_idx']])  # remove support
        self.label = None

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([transforms.Resize(size=(400, 400)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(self.img_mean, self.img_std)])

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):

        img_path = self.image_dirs[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = np.stack(3 * [img], axis=1)
        img = np.array([self.transform(Image.fromarray(i.transpose(1, 2, 0).astype(np.uint8))) for i in img])

        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))
        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        lbl = 1 * (lbl == self.label)

        sample = {'id': img_path}

        # Evaluation protocol.
        idx = lbl.sum(axis=(1, 2)) > 0

        img_idx_idx_ = [np.array(x, dtype=np.float32) for x in img[idx]]
        query_img = torch.stack([torch.from_numpy(x) for x in img_idx_idx_])

        query_mask = torch.from_numpy(lbl[idx])
        query_mask = F.interpolate(query_mask.unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        sample['image'] = query_img
        sample['label'] =query_mask

        return sample

    def get_support_index(self, n_shot, C):
        """
        Selecting intervals according to Ouyang et al.
        """
        if n_shot == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (n_shot * 2)
            part_interval = (1.0 - 1.0 / n_shot) / (n_shot - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_shot)]

        return (np.array(pcts) * C).astype('int')

    def getSupport(self, label=None, all_slices=True, N=None):
        if label is None:
            raise ValueError('Need to specify label class!')

        img_path = self.support_dir
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))  # [19, 1, 256, 256]
        img = np.stack(3 * [img], axis=1)  # [19, 3, 256, 256]
        img = np.array([self.transform(Image.fromarray(i.transpose(1, 2, 0).astype(np.uint8))) for i in img])

        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))
        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        lbl = 1 * (lbl == label)

        sample = {}
        if all_slices:
            support_img = torch.from_numpy(img)
            support_mask = torch.from_numpy(lbl)
            support_mask = torch.stack([F.interpolate(smask.float(), support_img.size()[-2:], mode='nearest')
                                        for smask in support_mask])

        else:
            # select N labeled slices
            if N is None:
                raise ValueError('Need to specify number of labeled slices!')
            idx = lbl.sum(axis=(1, 2)) > 0
            idx_ = self.get_support_index(N, idx.sum())

            img_idx_idx_ = [np.array(x, dtype=np.float32) for x in img[idx][idx_]]
            support_img = torch.stack([torch.from_numpy(x) for x in img_idx_idx_])

            support_mask = torch.from_numpy(lbl[idx][idx_])
            support_mask = torch.stack([F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_img.size()[-2:], mode='nearest').squeeze()
                                        for smask in support_mask])

        sample['image'] = support_img
        sample['label'] = support_mask

        return sample