"""
KITTI Road Dataset
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    import torchvision.transforms as T


class KITTIRoadDataset(Dataset):
    """KITTI Road 数据集"""
    
    def __init__(self, data_root, split='train', category='all',
                 img_h=384, img_w=1248, use_augmentation=True,
                 split_file=None):
        super().__init__()
        
        self.data_root = data_root
        self.split = split
        self.category = category
        self.img_h = img_h
        self.img_w = img_w
        self.use_augmentation = use_augmentation and split == 'train'
        self.split_file = split_file
        
        self.image_dir = os.path.join(data_root, 'training', 'image_2')
        self.adi_dir = os.path.join(data_root, 'training', 'ADI')
        self.label_dir = os.path.join(data_root, 'training', 'gt_image_2')
        
        self.samples = self._collect_samples()

        if split_file:
            with open(split_file, 'r', encoding='utf-8') as f:
                selected = {
                    line.strip().removesuffix('.png')
                    for line in f
                    if line.strip() and not line.lstrip().startswith('#')
                }
            sample_by_name = {sample['name']: sample for sample in self.samples}
            missing = sorted(selected - set(sample_by_name))
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} entries from split file are unavailable; "
                    f"first missing entries: {missing[:5]}"
                )
            self.samples = [sample_by_name[name] for name in sorted(selected)]
        elif split == 'train':
            self.samples = self.samples[:int(len(self.samples) * 0.8)]
        else:
            self.samples = self.samples[int(len(self.samples) * 0.8):]
        
        self.transform = self._get_transform()
        print(f"[{split}] 加载了 {len(self.samples)} 个样本 (类别: {category})")
    
    def _collect_samples(self):
        samples = []
        if not os.path.exists(self.image_dir):
            return samples
        
        for img_name in sorted(os.listdir(self.image_dir)):
            if not img_name.endswith('.png'):
                continue
            
            prefix = img_name.split('_')[0]
            if self.category != 'all' and prefix != self.category:
                continue
            
            base_name = img_name.replace('.png', '')
            rgb_path = os.path.join(self.image_dir, img_name)
            adi_path = os.path.join(self.adi_dir, img_name)
            label_path = os.path.join(self.label_dir, f"{prefix}_road_{base_name.split('_', 1)[1]}.png")
            
            if os.path.exists(rgb_path) and os.path.exists(adi_path) and os.path.exists(label_path):
                samples.append({
                    'name': base_name,
                    'rgb': rgb_path,
                    'adi': adi_path,
                    'label': label_path
                })
        
        return samples
    
    def _get_transform(self):
        if HAS_ALBUMENTATIONS:
            if self.use_augmentation:
                return A.Compose([
                    A.Resize(self.img_h, self.img_w),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], additional_targets={'adi': 'image'})
            else:
                return A.Compose([
                    A.Resize(self.img_h, self.img_w),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], additional_targets={'adi': 'image'})
        else:
            return T.Compose([
                T.Resize((self.img_h, self.img_w)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        rgb = np.array(Image.open(sample['rgb']).convert('RGB'))
        adi = np.array(Image.open(sample['adi']).convert('RGB'))
        label = np.array(Image.open(sample['label']))
        
        if label.ndim == 3:
            road_mask = (label[:, :, 2] > 200) & (label[:, :, 0] > 200)
            label = road_mask.astype(np.uint8)
        else:
            label = (label > 0).astype(np.uint8)
        
        if HAS_ALBUMENTATIONS:
            transformed = self.transform(image=rgb, adi=adi, mask=label)
            rgb = transformed['image']
            adi = transformed['adi']
            label = transformed['mask']
        else:
            rgb = self.transform(Image.fromarray(rgb))
            adi = self.transform(Image.fromarray(adi))
            label = torch.from_numpy(
                np.array(Image.fromarray(label).resize((self.img_w, self.img_h), Image.NEAREST))
            )
        
        return {
            'rgb': rgb,
            'adi': adi,
            'label': label.long() if isinstance(label, torch.Tensor) else torch.tensor(label).long(),
            'name': sample['name']
        }


class SyntheticDataset(Dataset):
    """合成数据集"""
    
    def __init__(self, num_samples=100, img_h=384, img_w=1248):
        self.num_samples = num_samples
        self.img_h = img_h
        self.img_w = img_w
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'rgb': torch.randn(3, self.img_h, self.img_w),
            'adi': torch.randn(3, self.img_h, self.img_w),
            'label': torch.randint(0, 2, (self.img_h, self.img_w)),
            'name': f'synthetic_{idx:06d}'
        }


def _seed_worker(worker_id):
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(data_root, split='train', category='all', batch_size=4,
                   num_workers=4, img_h=384, img_w=1248, use_augmentation=True,
                   use_synthetic=False, shuffle=None, split_file=None,
                   seed=42, drop_last=None):
    """获取数据加载器"""
    
    if use_synthetic:
        dataset = SyntheticDataset(
            num_samples=200 if split == 'train' else 50,
            img_h=img_h, img_w=img_w
        )
    else:
        dataset = KITTIRoadDataset(
            data_root=data_root, split=split, category=category,
            img_h=img_h, img_w=img_w, use_augmentation=use_augmentation,
            split_file=split_file,
        )
    
    if shuffle is None:
        shuffle = (split == 'train')
    if drop_last is None:
        # Preserve the historical public-loader behavior for callers that do
        # not opt into an explicit revision protocol.
        drop_last = (split == 'train')
    
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=drop_last,
        worker_init_fn=_seed_worker, generator=generator,
    )
