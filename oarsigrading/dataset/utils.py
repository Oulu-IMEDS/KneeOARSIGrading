import os
import glob
import pandas as pd
import cv2
import numpy as np
import torch

from torch.utils.data.sampler import Sampler
from torch._six import int_classes as _int_classes

from oarsigrading.dataset.metadata.oai import get_oai_meta
from oarsigrading.dataset.metadata.most import get_most_meta


def read_gs(fpath):
    return cv2.imread(fpath)


def build_dataset_meta(args, img_dir_name='MOST_OAI_FULL_0_2'):
    img_paths = glob.glob(os.path.join(args.dataset_root, img_dir_name, '*.png'))

    patient_ids = list(map(lambda x: x.split('/')[-1].split('_')[0], img_paths))
    files_metadata = pd.DataFrame(data={'fname': img_paths, 'ID': patient_ids})
    files_metadata['DS'] = files_metadata.apply(lambda x: 'MOST' if str(x[1]).startswith('M') else 'OAI', 1)
    files_metadata['VISIT'] = files_metadata.apply(lambda x: x[0].split('/')[-1].split('_')[1], 1)  # Follow up

    files_metadata_oai = files_metadata[files_metadata['DS'] == 'OAI']
    files_metadata_oai['SIDE'] = files_metadata_oai.apply(lambda x: 1 if x[0].
                                                          split('/')[-1].
                                                          split('_')[-1][:-4] == 'R' else 2, 1)

    files_metadata_most = files_metadata[files_metadata['DS'] == 'MOST']
    files_metadata_most['SIDE'] = files_metadata_most.apply(lambda x: 1 if x[0].
                                                            split('/')[-1].
                                                            split('_')[-1][:-4] == 'L' else 2, 1)
    oai_meta = get_oai_meta(os.path.join(args.meta_root, 'Data', 'metadata', 'OAI_meta'))
    most_meta = get_most_meta(os.path.join(args.meta_root, 'Data', 'metadata', 'MOST_meta'))
    common_cols = oai_meta.columns.intersection(most_meta.columns)

    oai_meta = pd.merge(oai_meta[common_cols], files_metadata_oai, on=('ID', 'SIDE', 'VISIT'))
    most_meta = pd.merge(most_meta[common_cols], files_metadata_most, on=('ID', 'SIDE', 'VISIT'))
    return oai_meta, most_meta


def make_weights_for_multilabel(weights_set, targets):
    weights_labels = np.log(weights_set.shape[0] / weights_set.sum(0))
    weights = np.zeros(targets.shape[0])

    for i in range(targets.shape[0]):
        weights[i] = weights_labels[np.where(targets[i])[0]].sum()

    return weights, weights_labels


class WeightedRandomSampler(Sampler):
    r"""Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Fixed version of PyTorch sampler.

    Arguments:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
    """

    def __init__(self, weights, num_samples, replacement=True):
        super(WeightedRandomSampler, self).__init__(weights)
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = weights.clone().detach().double()
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples
