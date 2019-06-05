import glob
import json
import numpy as np
import os
import argparse
from sklearn.metrics import classification_report, mean_squared_error, balanced_accuracy_score, cohen_kappa_score


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
                for model in ['resnet18', 'resnet34', 'resnet50', 'se_resnet50', 'se_resnext50_32x4d']:
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
                        f1_weighted = float(clf_rep.split('\n')[-2].split()[-2])

                        mse = mean_squared_error(gt[:, 0], predicts_kl.argmax(1))

                        acc = balanced_accuracy_score(gt[:, 0],
                                                      predicts_kl.argmax(1))

                        kappa = cohen_kappa_score(gt[:, 0], predicts_kl.argmax(1), weights='quadratic')

                        print(f'{np.round(f1_weighted, args.precision)} & '
                              f'{np.round(mse, args.precision)} & '
                              f'{np.round(acc, args.precision)} & '
                              f'{np.round(kappa, args.precision)} \\\\')

                        features = ['OARSI OST-TL', 'OARSI OST-FL', 'OARSI JSN-L',
                                    'OARSI OST-TM', 'OARSI OST-FM', 'OARSI JSN-M']

                        for feature_id, feature_name in enumerate(features):
                            clf_rep = classification_report(gt[:, feature_id+1],
                                                            predicts_oarsi[:, feature_id, :].argmax(1))
                            f1_weighted = float(clf_rep.split('\n')[-2].split()[-2])

                            mse = mean_squared_error(gt[:, feature_id+1], predicts_oarsi[:, feature_id, :].argmax(1))

                            acc = balanced_accuracy_score(gt[:, feature_id+1],
                                                          predicts_oarsi[:, feature_id, :].argmax(1))
                            kappa = cohen_kappa_score(gt[:, feature_id + 1], predicts_oarsi[:, feature_id, :].argmax(1),
                                                      weights='quadratic')

                            print(f'=======> ' + feature_name)
                            #print('F1-weighted', f1_weighted)
                            #print('MSE', np.round(mse, args.precision))
                            #print('Balanced acc.', np.round(acc, args.precision))
                            #print('Kappa', np.round(kappa, 2))

                            print(f'{np.round(f1_weighted, args.precision)} & '
                                  f'{np.round(mse, args.precision)} & '
                                  f'{np.round(acc, args.precision)} & '
                                  f'{np.round(kappa, args.precision)} \\\\')
