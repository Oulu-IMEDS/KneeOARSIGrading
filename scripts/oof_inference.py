import cv2
import sys
import pickle
import argparse
import os
import glob
import json

from termcolor import colored
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm


from oarsigrading.kvs import GlobalKVS
from oarsigrading.training.dataset import OARSIGradingDataset
import numpy as np
from oarsigrading.evaluation import metrics

from oarsigrading.training.model_zoo import backbone_name
from oarsigrading.training.model import OARSIGradingNet

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

DEBUG = sys.gettrace() is not None

if __name__ == "__main__":
    kvs = GlobalKVS()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='')
    parser.add_argument('--meta_root', default='')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--save_dir', default='')
    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_backup = pickle.load(f)

    train_set = session_backup["args"][0].train_set
    metadata = session_backup[f'{train_set}_meta'][0]

    layers, dw, se = session_backup['args'][0].backbone_depth, \
        session_backup['args'][0].dw, \
        session_backup['args'][0].se

    bb_name = backbone_name(layers, se, dw)
    print(colored('====> ', 'blue') + f'[{args.snapshot}] {bb_name}')
    predicts = []
    fnames = []
    gt = []
    for fold_id in range(session_backup['args'][0].n_folds):
        print(colored('====> ', 'red') + f'Loading fold [{fold_id}]')
        snapshot_name = glob.glob(os.path.join(args.snapshots_root, args.snapshot, f'fold_{fold_id}*.pth'))
        if len(snapshot_name) == 0:
            continue
        snapshot_name = snapshot_name[0]
        net = OARSIGradingNet(bb_depth=layers, dropout=session_backup['args'][0].dropout_rate,
                              cls_bnorm=session_backup['args'][0].use_bnorm, se=se, dw=dw,
                              use_gwap=getattr(session_backup['args'][0], 'use_gwap', False),
                              use_gwap_hidden=getattr(session_backup['args'][0], 'use_gwap_hidden', False))

        net.load_state_dict(torch.load(snapshot_name)['net'])

        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net).to('cuda')
        net.eval()

        _, val_index = session_backup['cv_split_all_folds'][0][fold_id]
        val_set = metadata.iloc[val_index]

        val_dataset = OARSIGradingDataset(val_set, session_backup['val_trf'][0])
        val_loader = DataLoader(val_dataset, batch_size=args.bs,
                                num_workers=args.n_threads,
                                sampler=SequentialSampler(val_dataset))

        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader), desc=f'Predicting fold {fold_id}:'):
                inputs = batch['img'].squeeze().to('cuda')
                outputs = net(inputs)
                tmp_preds = np.zeros(batch['target'].squeeze().size(), dtype=np.int64)
                for task_id, o in enumerate(outputs):
                    tmp_preds[:, task_id] = outputs[task_id].to('cpu').squeeze().argmax(1)

                predicts.append(tmp_preds)
                gt.append(batch['target'].to('cpu').numpy().squeeze())
                fnames.extend(batch['ID'])

    gt, predicts = np.vstack(gt).squeeze(), np.vstack(predicts)

    save_fld = os.path.join(args.snapshots_root, args.snapshot, 'oof_inference')
    os.makedirs(save_fld, exist_ok=True)
    np.savez_compressed(os.path.join(save_fld, 'results.npz'),
                        fnames=fnames,
                        gt=gt,
                        predicts=predicts)

    metrics_dict = metrics.compute_metrics(gt, predicts)
    model_info = dict()
    model_info['backbone'] = bb_name
    model_info['gwap'] = getattr(session_backup['args'][0], 'use_gwap', False)
    model_info['gwap_hidden'] = getattr(session_backup['args'][0], 'use_gwap_hidden', False)
    model_info['weighted_sampling'] = getattr(session_backup['args'][0], 'weighted_sampling', False)

    metrics_dict['model'] = model_info

    with open(os.path.join(save_fld, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f)
