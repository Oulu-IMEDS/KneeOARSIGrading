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
    parser.add_argument('--precision', type=int, default=4)
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
                        print(f'======> KL')
                        print(classification_report(gt[:, 0],
                                                    predicts_kl.argmax(1)))
                        print('MSE', np.round(mean_squared_error(gt[:, 0],
                                                                 predicts_kl.argmax(1)), 2))
                        print('Balanced acc.', np.round(balanced_accuracy_score(gt[:, 0],
                                                                                predicts_kl.argmax(1)), 2))
                        print('Kappa', np.round(cohen_kappa_score(gt[:, 0],
                                                                  predicts_kl.argmax(1), weights='quadratic'), 2))

                        print(f'======> OARSI OST-TL')
                        print(classification_report(gt[:, 1],
                                                    predicts_oarsi[:, 0, :].argmax(1)))
                        print('MSE', np.round(mean_squared_error(gt[:, 1],
                                                                 predicts_oarsi[:, 0, :].argmax(1)), 2))
                        print('Balanced acc.', np.round(balanced_accuracy_score(gt[:, 1],
                                                                                predicts_oarsi[:, 0, :].argmax(1)), 2))
                        print('Kappa', np.round(cohen_kappa_score(gt[:, 1],
                                                                  predicts_oarsi[:, 0, :].argmax(1),
                                                                  weights='quadratic'), 2))

                        print(f'======> OARSI OST-FL')
                        print(classification_report(gt[:, 2],
                                                    predicts_oarsi[:, 1, :].argmax(1)))
                        print('MSE', np.round(mean_squared_error(gt[:, 2],
                                                                 predicts_oarsi[:, 1, :].argmax(1)), 2))
                        print('Balanced acc.', np.round(balanced_accuracy_score(gt[:, 2],
                                                                                predicts_oarsi[:, 1, :].argmax(1)), 2))
                        print('Kappa', np.round(cohen_kappa_score(gt[:, 2],
                                                                  predicts_oarsi[:, 1, :].argmax(1),
                                                                  weights='quadratic'), 2))

                        print(f'======> OARSI JSN-L')
                        print(classification_report(gt[:, 3], predicts_oarsi[:, 2, :].argmax(1)))
                        print('MSE', np.round(mean_squared_error(gt[:, 3],
                                                                 predicts_oarsi[:, 2, :].argmax(1)), 2))
                        print('Balanced acc.', np.round(balanced_accuracy_score(gt[:, 3],
                                                                                predicts_oarsi[:, 2, :].argmax(1)), 2))
                        print('Kappa', np.round(cohen_kappa_score(gt[:, 3],
                                                                  predicts_oarsi[:, 2, :].argmax(1),
                                                                  weights='quadratic'), 2))

                        print(f'======> OARSI OST-TM')
                        print(classification_report(gt[:, 4], predicts_oarsi[:, 3, :].argmax(1)))
                        print('MSE', np.round(mean_squared_error(gt[:, 4],
                                                                 predicts_oarsi[:, 3, :].argmax(1)), 2))
                        print('Balanced acc.', np.round(balanced_accuracy_score(gt[:, 4],
                                                                                predicts_oarsi[:, 3, :].argmax(1)), 2))
                        print('Kappa', np.round(cohen_kappa_score(gt[:, 4],
                                                                  predicts_oarsi[:, 3, :].argmax(1),
                                                                  weights='quadratic'), 2))

                        print(f'======> OARSI OST-FM')
                        print(classification_report(gt[:, 5], predicts_oarsi[:, 4, :].argmax(1)))
                        print('MSE', np.round(mean_squared_error(gt[:, 5],
                                                                 predicts_oarsi[:, 4, :].argmax(1)), 2))
                        print('Balanced acc.', np.round(balanced_accuracy_score(gt[:, 5],
                                                                                predicts_oarsi[:, 4, :].argmax(1)), 2))
                        print('Kappa', np.round(cohen_kappa_score(gt[:, 5],
                                                                  predicts_oarsi[:, 4, :].argmax(1),
                                                                  weights='quadratic'), 2))

                        print(f'======> OARSI JSN-M')
                        print(classification_report(gt[:, 6],
                                                    predicts_oarsi[:, 5, :].argmax(1)))
                        print('MSE', np.round(mean_squared_error(gt[:, 6],
                                                                 predicts_oarsi[:, 5, :].argmax(1)), 2))
                        print('Balanced acc.', np.round(balanced_accuracy_score(gt[:, 6],
                                                                                predicts_oarsi[:, 5, :].argmax(1)), 2))
                        print('Kappa', np.round(cohen_kappa_score(gt[:, 6],
                                                                  predicts_oarsi[:, 5, :].argmax(1),
                                                                  weights='quadratic'), 2))
