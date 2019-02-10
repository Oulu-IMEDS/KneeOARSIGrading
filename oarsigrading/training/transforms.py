import copy
from functools import partial
import numpy as np
import solt.core as slc
import solt.transforms as slt
from torchvision import transforms
import torch

from oarsigrading.kvs import GlobalKVS

import cv2
import solt.data as sld


def wrap2solt(inp_data):
    img, entry = inp_data
    lndm_t = entry.landmarks_T.copy()
    lndm_f = entry.landmarks_F.copy()
    if entry.SIDE == 1:  # Left
        img = cv2.flip(img, 1)
        lndm_t[:, 0] = img.shape[1] - lndm_t[:, 0]
        lndm_f[:, 0] = img.shape[1] - lndm_f[:, 0]

        # Reversed OARSI grades (because of flipping)
        data_c_content = (img,
                          sld.KeyPoints(lndm_t, H=img.shape[0], W=img.shape[1]),  # Tibial landmarks
                          sld.KeyPoints(lndm_f, H=img.shape[0], W=img.shape[1]),  # Femoral landmarks
                          entry.XROSTM, entry.XROSFM, entry.XRJSM, entry.XROSTL, entry.XROSFL, entry.XRJSL)
    else:
        data_c_content = (img,
                          sld.KeyPoints(lndm_t, H=img.shape[0], W=img.shape[1]),
                          sld.KeyPoints(lndm_f, H=img.shape[0], W=img.shape[1]),
                          entry.XROSTL, entry.XROSFL, entry.XRJSL, entry.XROSTM, entry.XROSFM, entry.XRJSM)

    dc = sld.DataContainer(data_c_content, 'IPPLLLLLL')

    return dc


def unpack_solt_data(dc: sld.DataContainer):
    return dc.data


def apply_by_index(items, transform, idx=0):
    """Applies callable to certain objects in iterable using given indices.
    Parameters
    ----------
    items: tuple or list
    transform: callable
    idx: int or tuple or or list None
    Returns
    -------
    result: tuple
    """
    if idx is None:
        return items
    if not isinstance(items, (tuple, list)):
        raise TypeError
    if not isinstance(idx, (int, tuple, list)):
        raise TypeError

    if isinstance(idx, int):
        idx = [idx, ]

    idx = set(idx)
    res = []
    for i, item in enumerate(items):
        if i in idx:
            res.append(transform(item))
        else:
            res.append(copy.deepcopy(item))

    return res


def normalize_channel_wise(tensor, mean, std):
    if len(tensor.size()) != 3:
        raise ValueError

    for channel in range(tensor.size(0)):
        tensor[channel, :, :] -= mean[channel]
        tensor[channel, :, :] /= std[channel]

    return tensor


def pack_tensors(res):
    img_res, lndm_t, lndm_f, ostl, osfl, jsl, ostm, osfm, osjsm = res

    img_res = transforms.ToTensor()(img_res)
    grades = torch.FloatTensor(np.round([ostl, osfl, jsl, ostm, osfm, osjsm]).astype(int)).unsqueeze(0)

    return img_res, grades


def init_transforms(mean_vector, std_vector):
    kvs = GlobalKVS()

    if mean_vector is not None:
        mean_vector = torch.from_numpy(mean_vector).float()
        std_vector = torch.from_numpy(std_vector).float()
        norm_trf = partial(normalize_channel_wise, mean=mean_vector, std=std_vector)
    else:
        norm_trf = None

    train_trf = [
        wrap2solt,
        slc.Stream([
            slt.RandomRotate(rotation_range=(-5, 5), interpolation='bilinear', p=0.8),
            slt.CropTransform(kvs['args'].imsize, crop_mode='c'),
            slt.CropTransform(kvs['args'].crop_size, crop_mode='r'),  # 130mm (resolution 0.2mm)
            slt.ResizeTransform(350),
            slt.ImageGammaCorrection(p=1, gamma_range=(0.5, 2.5))
        ]),
        unpack_solt_data,
        pack_tensors,
    ]

    val_trf = [
        wrap2solt,
        slc.Stream([
            slt.CropTransform(kvs['args'].crop_size, crop_mode='c'),  # 130mm (resolution 0.2mm)
            slt.ResizeTransform(350),
        ]),
        unpack_solt_data,
        pack_tensors,
        norm_trf,
    ]

    if norm_trf is not None:
        train_trf.append(norm_trf)
        val_trf.append(norm_trf)

    train_trf = transforms.Compose(train_trf)
    val_trf = transforms.Compose(val_trf)

    return train_trf, val_trf
