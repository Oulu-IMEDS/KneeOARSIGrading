import numpy as np
import cv2
import pandas as pd
from sas7bdat import SAS7BDAT


def load_landmarks(img_fname, landmarks, sizepx, size_max=6000):
    ID, FU, SIDE = img_fname.split('/')[-1].split('.')[0].split('_')

    try:
        lndm = landmarks.loc[f'{ID}_{FU}']
    except:
        return None, None
    landmarks_t = lndm.landmarks[f'T{SIDE}']
    landmarks_f = lndm.landmarks[f'F{SIDE}']

    N_t = landmarks_t.shape[0]
    N_f = landmarks_f.shape[0]

    if SIDE == 'L':
        landmarks_t[:, 0] = size_max - landmarks_t[:, 0]
        landmarks_f[:, 0] = size_max - landmarks_f[:, 0]

    cx, cy = landmarks_t[N_t // 2, :].astype(int)
    p1, p2 = landmarks_t[0], landmarks_t[-1]
    ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

    M = cv2.getRotationMatrix2D((cx, cy), ang, 1)

    landmarks_t = np.dot(M, np.hstack((landmarks_t, np.ones((N_t, 1)))).T).T[:, [0, 1]]
    landmarks_f = np.dot(M, np.hstack((landmarks_f, np.ones((N_f, 1)))).T).T[:, [0, 1]]

    landmarks_t = landmarks_t - (cx, cy) + (sizepx // 2, sizepx // 2)
    landmarks_f = landmarks_f - (cx, cy) + (sizepx // 2, sizepx // 2)

    if SIDE == 'L':
        img = cv2.imread(img_fname, 0)
        landmarks_t[:, 0] = img.shape[1] - landmarks_t[:, 0]
        landmarks_f[:, 0] = img.shape[1] - landmarks_f[:, 0]

    return landmarks_t, landmarks_f


def read_sas7bdat(fpath):
    rows = []
    with SAS7BDAT(fpath) as f:
        for row in f:
            rows.append(row)
    return pd.DataFrame(rows[1:], columns=rows[0])
