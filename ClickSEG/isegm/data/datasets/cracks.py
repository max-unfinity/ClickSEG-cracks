from pathlib import Path
import pickle
import random
import numpy as np
import json
from copy import deepcopy
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from PIL import Image
import cv2
import os
import torch
 

class CracksDataset(ISDataset):
    def __init__(self, dataset_path, split='train', n_test_samples=50, use_morph=True, **kwargs):
        super().__init__(**kwargs)
        self.name = 'Cracks'
        self.split = split
        self.use_morph = use_morph
        dataset_path = Path(dataset_path)
        self._split_path = dataset_path / split
        self._images_path = self._split_path / 'images'
        self._masks_path = self._split_path / 'masks'
        
        cache_path = split+'_files.pt'
        if os.path.exists(cache_path):
            print('using cached files')
            self.img_files, self.mask_files = torch.load(cache_path)
        else:
            self.img_files = list(map(str, self._images_path.rglob('*.jpg')))
            self.mask_files = list(map(str, self._masks_path.rglob('*.jpg')))
            torch.save((self.img_files, self.mask_files), cache_path)
            
        if split == 'test':
            n = len(self.img_files)
            n_test_samples = min(n, n_test_samples)
            print(f'trunc test to {n_test_samples}')
            step = n//n_test_samples
            self.img_files, self.mask_files = self.img_files[:step*n_test_samples:step], self.mask_files[:step*n_test_samples:step]
        
        self.dataset_samples = self.img_files
        assert len(self.img_files) == len(self.mask_files)

    def get_sample(self, index) -> DSample:
        img = np.asarray(Image.open(self.img_files[index]))
        mask = np.asarray(Image.open(self.mask_files[index]).convert('L'))
        assert mask.ndim == 2
        
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        if self.use_morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = mask[:,:,None].astype(np.int32)
        return DSample(img, mask, objects_ids=[1], sample_id=index)
