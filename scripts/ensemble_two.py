import numpy as np
import argparse
import pickle
import os
from oarsigrading.evaluation import metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--snapshot1', default='')
    parser.add_argument('--snapshot2', default='')
    parser.add_argument('--tta1', type=bool, default=False)
    parser.add_argument('--tta2', type=bool, default=False)

    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot1, 'session.pkl'), 'rb') as f:
        session_1_backup = pickle.load(f)

    with open(os.path.join(args.snapshots_root, args.snapshot1, 'session.pkl'), 'rb') as f:
        session_2_backup = pickle.load(f)

    snapshot_1 = np.load(os.path.join(args.snapshots_root, args.snapshot1, 'oof_inference',
                                      f'results_{"TTA" if args.tta1 else "plain"}.npz'))

    snapshot_2 = np.load(os.path.join(args.snapshots_root, args.snapshot2, 'oof_inference',
                                      f'results_{"TTA" if args.tta2 else "plain"}.npz'))

    probs_kl = (snapshot_1['probs_kl'] + snapshot_2['probs_kl']) / 2
    probs_oarsi = (snapshot_1['probs_oarsi'] + snapshot_2['probs_oarsi']) / 2
    if snapshot_1['gt'].shape[1] < snapshot_2['gt'].shape[1]:
        gt = snapshot_2['gt']
    else:
        gt = snapshot_1['gt']

    predicts = list()
    predicts.append(np.expand_dims(probs_kl.squeeze().argmax(1), 1))

    for task_id in range(probs_oarsi.shape[1]):
        predicts.append(np.expand_dims(probs_oarsi[:, task_id].squeeze().argmax(1), 1))

    predicts = np.hstack(predicts)
    metrics_dict = metrics.compute_metrics(predicts, gt, no_kl=False)



