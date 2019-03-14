import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import glob
import json
import numpy as np
import os
import argparse
from sklearn.metrics import classification_report, mean_squared_error, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import roc_auc_score, roc_curve

model_dict = {'resnet18': 'Resnet-18',
             'resnet34': 'Resnet-34',
             'resnet50': 'Resnet-50',
             'se_resnet50': 'SE-Resnet-50',
             'se_resnext50_32x4d': 'SE-ResNext50-32x4d'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots_dir', default='/media/lext/FAST/OARSI_grading_project/workdir/'
                                                   'oarsi_grades_snapshots_weighted/')
    parser.add_argument('--precision', type=int, default=2)
    args = parser.parse_args()

    for weighted in [False, True]:
        for gwap in [False, True]:
            for gwap_hidden in [False, True]:
                exp = ''
                exp += 'WS_' if weighted else 'NWS_'
                if gwap:
                    exp += 'h-GWAP' if gwap_hidden else 'GWAP'
                else:
                    exp += 'no-GWAP'
                print(exp)
                print('='*80)
                for model in ['se_resnext50_32x4d']:
                    for snp in glob.glob(os.path.join(args.snapshots_dir, '*', 'test_inference',
                                                      'results_plain.npz')):

                        with open(os.path.join(snp.split('results_plain.npz')[0], 'metrics_plain.json')) as f:
                            test_res = json.load(f)
                            
                        if test_res['model']['backbone'] != model:
                            continue
                        if test_res['model']['weighted_sampling'] != weighted:
                            continue
                        if test_res['model']['gwap'] != gwap:
                            continue
                        if test_res['model']['gwap_hidden'] != gwap_hidden:
                            continue
                        
                        data = np.load(snp)
                        gt = data['gt']

                        predicts_kl = data['predicts_kl']
                        predicts_oarsi = data['predicts_oarsi']

                        print(f'====> {model}')
                        print(f'=======> KL')
                        clf_rep = classification_report(gt[:, 0],
                                                        predicts_kl.argmax(1))

                        probs_oa = predicts_kl[:, 2:].sum(1)
                        gt_oa = gt[:, 0] >= 2

                        features = ['OARSI OST-TL', 'OARSI OST-FL', 'OARSI JSN-L',
                                    'OARSI OST-TM', 'OARSI OST-FM', 'OARSI JSN-M']

                        fpr1, tpr1, _ = roc_curve(gt_oa, probs_oa)
                        auc = np.round(roc_auc_score(gt_oa,  probs_oa), 4)

                        matplotlib.rcParams.update({'font.size': 16})
                        plt.figure(figsize=(6, 6))
                        plt.plot(fpr1, tpr1, label='OA vs non-OA'.format(probs_oa.shape[0]), lw=2, c='b')
                        plt.grid()
                        plt.xlabel('False positive rate')
                        plt.ylabel('True positive rate')
                        plt.xlim(0, 1)
                        plt.ylim(0, 1)
                        plt.tight_layout()
                        plt.show()

                        print('AUC [OA]:', auc)



