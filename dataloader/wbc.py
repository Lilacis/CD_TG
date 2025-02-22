import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import PIL.Image as Image
from torchvision import transforms

class DatasetWBC(Dataset):
    def __init__(self, base_path, class_name, image_type,shot=1):
        """
        :param base_path: dataset root
        :param class_name: class name 
        :param shot: shot of support image
        """
        self.shot = shot
        self.class_dir = os.path.join(base_path, class_name)
        self.benchmark = 'WBC'
        self.class_ids = range(0,2)
        
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([transforms.Resize(size=(321, 321)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.img_mean, self.img_std)])
        self.type = image_type

        self.img_metadata = self.build_img_metadata(type=self.type)
        print(f"Found {len(self.img_metadata)} images in {self.class_dir}")

    def __len__(self):
        return len(self.img_metadata)

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def __getitem__(self, idx):
        query_name = self.img_metadata[idx]
        support_names = self.sample_support_names(query_name)

        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        support_masks = torch.stack([
            F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            for smask in support_masks
        ])

        batch = {
            'query_img': query_img,
            'query_mask': query_mask,
            'query_name': query_name,

            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'support_names': support_names
        }

        return batch

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]
    
        query_mask_name = query_name.replace(f'.{self.type}', '_mask.png')
        query_mask = self.read_mask(query_mask_name)
    
        support_masks = [
            self.read_mask(name.replace(f'.{self.type}', '_mask.png'))
            for name in support_names
        ]
    
        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, mask_path):
        mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_support_names(self, query_name):
        support_names = []
        while len(support_names) < self.shot:
            support_name = np.random.choice(self.img_metadata, 1, replace=False)[0]
            if query_name != support_name:
                support_names.append(support_name)
        return support_names

    def build_img_metadata(self, type='png'):
        img_metadata = sorted(glob.glob(os.path.join(self.class_dir, f'*.{type}')))
        return [img for img in img_metadata if not os.path.basename(img).endswith('_mask.png')]


def create_loaders(base_path, shot=1, batch_size=1, class_names=None, image_type='png'):    
    loaders = {}

    for class_name in class_names:
        dataset = DatasetWBC(base_path, class_name, shot=shot, image_type=image_type)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        loaders[class_name] = loader

    return loaders
