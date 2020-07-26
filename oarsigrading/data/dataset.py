import torch
from torch.utils import data
import cv2
import numpy as np
from oarsigrading.data.utils import read_gs


class OARSIGradingDataset(data.Dataset):
    def __init__(self, cfg, metadata, transforms):
        super(OARSIGradingDataset, self).__init__()

        self.meta = metadata
        self.cfg = cfg
        self.trf = transforms

    def __getitem__(self, idx):
        entry = self.meta.iloc[idx]
        img = read_gs(entry.fname)
        if entry.SIDE == 1:  # Right looks like left
            img = cv2.flip(img, 1)

        labels = [entry.XRKL, entry.XROSTL, entry.XROSFL, entry.XRJSL, entry.XROSTM, entry.XROSFM, entry.XRJSM]
        res_dc = self.trf({'image': img, 'labels': labels})
        grades = torch.from_numpy(np.round(res_dc['labels']).astype(int)).unsqueeze(0)
        if self.cfg.training.no_kl:
            grades = grades[:, 1:] # KL grade goes first
        return {'image': res_dc['image'], 'target': grades, 'ID': entry.ID,
                'SIDE': entry.SIDE, 'VISIT': entry.VISIT}

    def __len__(self):
        return self.meta.shape[0]
