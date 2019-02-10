import os
import glob
import pandas as pd
from tqdm import tqdm
import cv2

from oarsigrading.dataset.metadata.utils import load_landmarks
from oarsigrading.dataset.metadata.oai import get_oai_meta
from oarsigrading.dataset.metadata.most import get_most_meta


def read_gs(fpath):
    return cv2.imread(fpath, 0)


def build_dataset(args, img_dir_name='MOST_OAI_FULL_0_2'):
    img_paths = glob.glob(os.path.join(args.dataset_root, img_dir_name, '*.png'))

    patient_ids = list(map(lambda x: x.split('/')[-1].split('_')[0], img_paths))
    files_metadata = pd.DataFrame(data={'fname': img_paths, 'ID': patient_ids})
    files_metadata['DS'] = files_metadata.apply(lambda x: 'MOST' if str(x[1]).startswith('M') else 'OAI', 1)
    files_metadata['VISIT'] = files_metadata.apply(lambda x: x[0].split('/')[-1].split('_')[1], 1)  # Follow up
    files_metadata['SIDE'] = files_metadata.apply(lambda x: 1 if x[0].split('/')[-1].split('_')[-1][:-4] == 'L' else 2,
                                                  1)

    l_f = []
    l_t = []
    landmarks = pd.read_pickle(os.path.join(args.dataset_root, 'landmarks_scaled.pkl'))
    landmarks = landmarks.set_index('fname')

    for fname in tqdm(files_metadata.fname, total=files_metadata.shape[0], desc='Reading landmarks::'):
        res = load_landmarks(fname, landmarks, 700)
        l_t.append(res[0])
        l_f.append(res[1])

    files_metadata['landmarks_T'] = pd.Series(l_t)
    files_metadata['landmarks_F'] = pd.Series(l_f)

    files_metadata_oai = files_metadata[files_metadata['DS'] == 'OAI']
    files_metadata_most = files_metadata[files_metadata['DS'] == 'MOST']

    oai_meta = get_oai_meta(os.path.join(args.meta_root, 'Data', 'metadata', 'OAI_meta'))
    most_meta = get_most_meta(os.path.join(args.meta_root, 'Data', 'metadata', 'MOST_meta'))
    common_cols = oai_meta.columns.intersection(most_meta.columns)

    oai_meta = pd.merge(oai_meta[common_cols], files_metadata_oai, on=('ID', 'SIDE', 'VISIT'))
    most_meta = pd.merge(most_meta[common_cols], files_metadata_most, on=('ID', 'SIDE', 'VISIT'))
    return oai_meta, most_meta
