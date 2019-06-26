import numpy as np
import argparse
import pickle
import os
from oarsigrading.evaluation import metrics
import json
from oarsigrading.training.model_zoo import backbone_name


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

    with open(os.path.join(args.snapshots_root, args.snapshot2, 'session.pkl'), 'rb') as f:
        session_2_backup = pickle.load(f)

    layers, dw, se = session_1_backup['args'][0].backbone_depth, \
        session_1_backup['args'][0].dw, \
        session_1_backup['args'][0].se
    bb_name_1 = backbone_name(layers, se, dw)

    layers, dw, se = session_2_backup['args'][0].backbone_depth, \
        session_2_backup['args'][0].dw, \
        session_2_backup['args'][0].se
    bb_name_2 = backbone_name(layers, se, dw)

    snapshot_1 = np.load(os.path.join(args.snapshots_root, args.snapshot1, 'oof_inference',
                                      f'results_{"TTA" if args.tta1 else "plain"}.npz'))

    snapshot_2 = np.load(os.path.join(args.snapshots_root, args.snapshot2, 'oof_inference',
                                      f'results_{"TTA" if args.tta2 else "plain"}.npz'))

    snapshot_1_test = np.load(os.path.join(args.snapshots_root, args.snapshot1, 'test_inference',
                                           f'results_{"TTA" if args.tta1 else "plain"}.npz'))

    snapshot_2_test = np.load(os.path.join(args.snapshots_root, args.snapshot2, 'test_inference',
                                           f'results_{"TTA" if args.tta2 else "plain"}.npz'))

    probs_kl = (snapshot_1['probs_kl'] + snapshot_2['probs_kl']) / 2
    probs_oarsi = (snapshot_1['probs_oarsi'] + snapshot_2['probs_oarsi']) / 2
    if snapshot_1['gt'].shape[1] < snapshot_2['gt'].shape[1]:
        gt = snapshot_2['gt']
        gt_test = snapshot_2_test['gt']
    else:
        gt = snapshot_1['gt']
        gt_test = snapshot_1_test['gt']

    predicts = list()
    predicts.append(np.expand_dims(probs_kl.squeeze().argmax(1), 1))

    for task_id in range(probs_oarsi.shape[1]):
        predicts.append(np.expand_dims(probs_oarsi[:, task_id].squeeze().argmax(1), 1))

    predicts = np.hstack(predicts)
    metrics.compute_metrics(predicts, gt, no_kl=False)

    print('=='*20)
    print('====> Test set predictions:')
    print('=='*20)

    save_fld = os.path.join(args.snapshots_root, f'ens_{args.snapshot1}_{args.snapshot2}', 'test_inference')
    os.makedirs(save_fld, exist_ok=True)

    probs_kl = (snapshot_1_test['predicts_kl'] + snapshot_2_test['predicts_kl']) / 2
    probs_oarsi = (snapshot_1_test['predicts_oarsi'] + snapshot_2_test['predicts_oarsi']) / 2

    np.savez_compressed(os.path.join(save_fld, f'results_plain.npz'),
                        ids=snapshot_1_test['ids'],
                        visits=snapshot_1_test['visits'],
                        sides=snapshot_1_test['sides'],
                        gt=gt_test,
                        predicts_oarsi=probs_oarsi,
                        predicts_kl=probs_kl)

    kl_preds = probs_kl.argmax(1)
    oarsi_preds = probs_oarsi.argmax(2)
    stacked = np.hstack((kl_preds.reshape(kl_preds.shape[0], 1), oarsi_preds))

    metrics_dict = metrics.compute_metrics(gt_test, stacked, no_kl=False)
    model_info = dict()
    model_info['backbone'] = f'ens_{bb_name_1}_{bb_name_2}'
    model_info['gwap'] = False
    model_info['gwap_hidden'] = False
    model_info['weighted_sampling'] = False
    metrics_dict['model'] = model_info

    with open(os.path.join(save_fld, f'metrics_plain.json'), 'w') as f:
        json.dump(metrics_dict, f)
