import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import glob
import json
import numpy as np
import os
import argparse
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

model_dict = {'resnet18': 'Resnet-18',
             'resnet34': 'Resnet-34',
             'resnet50': 'Resnet-50',
             'se_resnet50': 'SE-Resnet-50',
             'se_resnext50_32x4d': 'SE-ResNext50-32x4d'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots_dir', default='/media/lext/FAST/OARSI_grading_project/workdir/'
                                                   'oarsi_grades_snapshots_weighted/')
    parser.add_argument('--save_dir', default='/media/lext/FAST/OARSI_grading_project/workdir/Results')
    parser.add_argument('--precision', type=int, default=2)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.save_dir, 'pics'), exist_ok=True)

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

                        probs_oa = predicts_kl[:, 2:].sum(1)
                        gt_oa = gt[:, 0] >= 2

                        probs_ostl = np.hstack((predicts_oarsi[:, 0, 1:].sum(1), predicts_oarsi[:, 1, 1:].sum(1)))
                        gt_ostl = np.hstack((gt[:, 1] >= 1, gt[:, 2] >= 1))

                        probs_ostm = np.hstack((predicts_oarsi[:, 3, 1:].sum(1), predicts_oarsi[:, 4, 1:].sum(1)))
                        gt_ostm = np.hstack((gt[:, 4] >= 1, gt[:, 5] >= 1))

                        probs_jsnl = predicts_oarsi[:, 3, 1:].sum(1)
                        gt_jsnl = gt[:, 4] >= 1

                        probs_jsnm = predicts_oarsi[:, 5, 1:].sum(1)
                        gt_jsnm = gt[:, 6] >= 1

                        features = ['OARSI OST-TL', 'OARSI OST-FL', 'OARSI JSN-L',
                                    'OARSI OST-TM', 'OARSI OST-FM', 'OARSI JSN-M']

                        matplotlib.rcParams.update({'font.size': 18})
                        f = plt.figure(figsize=(6, 6))

                        fpr1, tpr1, _ = roc_curve(gt_oa, probs_oa)
                        auc_oa = np.round(roc_auc_score(gt_oa,  probs_oa), 4)
                        plt.plot(fpr1, tpr1, label=f'OA vs non-OA. AUC {auc_oa:.2}', lw=2, c='b')

                        fpr1, tpr1, _ = roc_curve(gt_ostl, probs_ostl)
                        auc_ostl = np.round(roc_auc_score(gt_ostl, probs_ostl), 4)
                        plt.plot(fpr1, tpr1, label=f'Ost. Lateral. AUC {auc_ostl:.2}', lw=2, c='r')

                        fpr1, tpr1, _ = roc_curve(gt_ostm, probs_ostm)
                        auc_ostm = np.round(roc_auc_score(gt_ostm, probs_ostm), 4)
                        plt.plot(fpr1, tpr1, label=f'Ost. Medial. AUC {auc_ostm:.2}', lw=2, c='r', linestyle='--')

                        fpr1, tpr1, _ = roc_curve(gt_jsnl, probs_jsnl)
                        auc_jsnl = np.round(roc_auc_score(gt_jsnl, probs_jsnl), 4)
                        plt.plot(fpr1, tpr1, label=f'JSN. Lateral. AUC {auc_jsnl:.2}', lw=2, c='g')

                        fpr1, tpr1, _ = roc_curve(gt_jsnm, probs_jsnm)
                        auc_jsnm = np.round(roc_auc_score(gt_jsnm, probs_jsnm), 4)
                        plt.plot(fpr1, tpr1, label=f'JSN. Medial. AUC {auc_jsnm:.2}', lw=2, c='g', linestyle='--')

                        plt.grid()
                        plt.xlabel('False positive rate')
                        plt.ylabel('True positive rate')
                        plt.xlim(0, 1)
                        plt.ylim(0, 1)
                        plt.tight_layout()
                        plt.legend()
                        plt.savefig(os.path.join(args.save_dir, 'pics', 'oa_roc.pdf'), bbox_inches='tight')
                        plt.close(f)

                        matplotlib.rcParams.update({'font.size': 18})
                        f = plt.figure(figsize=(6, 6))

                        recall, precision, _ = precision_recall_curve(gt_oa, probs_oa)
                        ap_oa = np.round(average_precision_score(gt_oa,  probs_oa), 4)
                        plt.plot(recall, precision, label=f'OA vs non-OA. AP {ap_oa:.2}', lw=2, c='b')

                        recall, precision, _ = precision_recall_curve(gt_ostl, probs_ostl)
                        ap_ostl = np.round(average_precision_score(gt_ostl, probs_ostl), 4)
                        plt.plot(recall, precision, label=f'Ost. Lateral. AP {ap_ostl:.2}', lw=2, c='r')

                        recall, precision, _ = precision_recall_curve(gt_ostm, probs_ostm)
                        ap_ostm = np.round(average_precision_score(gt_ostm, probs_ostm), 4)
                        plt.plot(recall, precision, label=f'Ost. Medial. AP {ap_ostm:.2}', lw=2, c='r', linestyle='--')

                        recall, precision, _ = precision_recall_curve(gt_jsnl, probs_jsnl)
                        ap_jsnl = np.round(average_precision_score(gt_jsnl, probs_jsnl), 4)
                        plt.plot(recall, precision, label=f'JSN. Lateral. AP {ap_jsnl:.2}', lw=2, c='g')

                        recall, precision, _ = precision_recall_curve(gt_jsnm, probs_jsnm)
                        ap_jsnm = np.round(average_precision_score(gt_jsnm, probs_jsnm), 4)
                        plt.plot(recall, precision, label=f'JSN. Medial. AP {ap_jsnm:.2}', lw=2, c='g', linestyle='--')

                        plt.grid()
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                        plt.xlim(0, 1)
                        plt.ylim(0, 1)
                        plt.tight_layout()
                        plt.legend()
                        plt.savefig(os.path.join(args.save_dir, 'pics', 'oa_pr.pdf'), bbox_inches='tight')
                        plt.show()



