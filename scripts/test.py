import cv2
import sys
import pickle
import argparse
import os
import glob
import json
from functools import partial
import numpy as np

from termcolor import colored
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torchvision.transforms import Compose
from tqdm import tqdm


from oarsigrading.kvs import GlobalKVS
from oarsigrading.training.dataset import OARSIGradingDataset
from oarsigrading.evaluation import metrics
from oarsigrading.training.model_zoo import backbone_name
from oarsigrading.training.model import OARSIGradingNet, OARSIGradingNetSiamese
import oarsigrading.evaluation.tta as tta
from oarsigrading.training.transforms import apply_by_index
import pandas as pd

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

DEBUG = sys.gettrace() is not None

if __name__ == "__main__":
    kvs = GlobalKVS()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='')
    parser.add_argument('--meta_root', default='')
    parser.add_argument('--tta', type=bool, default=False)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--save_dir', default='')
    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_backup = pickle.load(f)

    oai_meta = pd.read_pickle(os.path.join(args.snapshots_root, 'oai_meta.pkl'))
    most_meta = pd.read_pickle(os.path.join(args.snapshots_root, 'most_meta.pkl'))

    most_meta = most_meta[(most_meta.XRKL >= 0) & (most_meta.XRKL <= 4)]
    oai_meta = oai_meta[(oai_meta.XRKL >= 0) & (oai_meta.XRKL <= 4)]

    if session_backup["args"][0].train_set == 'oai':
        metadata = most_meta
    else:
        metadata = oai_meta
    metadata.fname = metadata.fname.apply(lambda x: x.replace('/media/lext/FAST/OARSI_grading_project/Data/datasets/',
                                                              args.dataset_root), 1)

    layers, dw, se = session_backup['args'][0].backbone_depth, \
        session_backup['args'][0].dw, \
        session_backup['args'][0].se

    bb_name = backbone_name(layers, se, dw)
    print(colored('====> ', 'blue') + f'[{args.snapshot}] {bb_name}')
    predicts_avg_oarsi = 0
    predicts_avg_kl = 0
    for fold_id in range(session_backup['args'][0].n_folds):
        
        predicts_oarsi = list()
        predicts_kl = list()
        gt = list()
        visits = list()
        sides = list()
        ids = list()
        
        print(colored('====> ', 'red') + f'Loading fold [{fold_id}]')
        snapshot_name = glob.glob(os.path.join(args.snapshots_root, args.snapshot, f'fold_{fold_id}*.pth'))
        if len(snapshot_name) == 0:
            continue
        snapshot_name = snapshot_name[0]
        if getattr(session_backup['args'][0], 'siamese', False):
            net = OARSIGradingNetSiamese(backbone=session_backup['args'][0].siamese_bb,
                                         dropout=session_backup['args'][0].dropout_rate)
        else:
            net = OARSIGradingNet(bb_depth=layers, dropout=session_backup['args'][0].dropout_rate,
                                  cls_bnorm=session_backup['args'][0].use_bnorm, se=se, dw=dw,
                                  use_gwap=getattr(session_backup['args'][0], 'use_gwap', False),
                                  use_gwap_hidden=getattr(session_backup['args'][0], 'use_gwap_hidden', False),
                                  no_kl=getattr(session_backup['args'][0], 'no_kl', False))

        net.load_state_dict(torch.load(snapshot_name)['net'])

        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net).to('cuda')
        net.eval()

        if args.tta:
            print(colored('====> ', 'green') + f'5-crop TTA will be used')
            tta_cropper = partial(apply_by_index,
                                  transform=partial(tta.five_crop, size=session_backup['args'][0].crop_size),
                                  idx=0)

            test_trf = Compose([session_backup['val_trf'][0], tta_cropper])
        else:
            test_trf = session_backup['val_trf'][0]

        test_dataset = OARSIGradingDataset(metadata, test_trf)
        val_loader = DataLoader(test_dataset, batch_size=args.bs,
                                num_workers=args.n_threads,
                                sampler=SequentialSampler(test_dataset))

        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader), desc=f'Predicting fold {fold_id}:'):
                if getattr(session_backup['args'][0], 'siamese', False):
                    inp_med = batch['img_med']
                    inp_lat = batch['img_lat']
                    tmp_preds = tta.eval_batch(net, (inp_med, inp_lat),
                                               batch['target'], return_probs=True)
                else:
                    tmp_preds = tta.eval_batch(net, batch['img'],
                                               batch['target'], return_probs=True)

                if isinstance(tmp_preds, tuple):
                    kl_probs, oarsi_probs = tmp_preds
                else:
                    oarsi_probs = tmp_preds
                    kl_probs = np.zeros((oarsi_probs.shape[0], 5))

                predicts_kl.append(kl_probs)
                predicts_oarsi.append(oarsi_probs)

                gt.append(batch['target'].to('cpu').numpy().squeeze())
                ids.extend(batch['ID'])
                visits.extend(batch['VISIT'])
                sides.extend(batch['SIDE'])
        
        gt, predicts_kl, predicts_oarsi = np.vstack(gt).squeeze(), np.vstack(predicts_kl), np.vstack(predicts_oarsi)
        predicts_avg_oarsi += predicts_oarsi
        predicts_avg_kl += predicts_kl

    predicts_avg_oarsi /= session_backup['args'][0].n_folds
    predicts_avg_kl /= session_backup['args'][0].n_folds

    save_fld = os.path.join(args.snapshots_root, args.snapshot, 'test_inference')
    os.makedirs(save_fld, exist_ok=True)
    np.savez_compressed(os.path.join(save_fld, f'results_{"TTA" if args.tta else "plain"}.npz'),
                        ids=ids,
                        sides=sides,
                        visits=visits,
                        gt=gt,
                        predicts_kl=predicts_avg_kl,
                        predicts_oarsi=predicts_avg_oarsi)

    kl_preds = predicts_avg_kl.argmax(1)
    oarsi_preds = predicts_avg_oarsi.argmax(2)
    stacked = np.hstack((kl_preds.reshape(kl_preds.shape[0], 1), oarsi_preds))

    if getattr(session_backup['args'][0], 'no_kl', False):
        stacked = stacked[:, 1:]
    metrics_dict = metrics.compute_metrics(gt, stacked, no_kl=getattr(session_backup['args'][0], 'no_kl', False))
    model_info = dict()
    model_info['backbone'] = bb_name
    model_info['gwap'] = getattr(session_backup['args'][0], 'use_gwap', False)
    model_info['gwap_hidden'] = getattr(session_backup['args'][0], 'use_gwap_hidden', False)
    model_info['weighted_sampling'] = getattr(session_backup['args'][0], 'weighted_sampling', False)

    metrics_dict['model'] = model_info

    with open(os.path.join(save_fld, f'metrics_{"TTA" if args.tta else "plain"}.json'), 'w') as f:
        json.dump(metrics_dict, f)
