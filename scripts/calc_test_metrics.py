import glob
import json
import numpy as np
import os
import argparse
from sklearn.metrics import classification_report, mean_squared_error, balanced_accuracy_score, cohen_kappa_score
from functools import partial
from oarsigrading.evaluation.metrics import bootstrap_ci, plot_confusion


model_dict = {'resnet18': 'Resnet-18',
              'resnet34': 'Resnet-34',
              'resnet50': 'Resnet-50',
              'se_resnet50': 'SE-Resnet-50',
              'se_resnext50_32x4d': 'SE-ResNext50-32x4d',
              'ens_se_resnet50_se_resnext50_32x4d': 'Ensemble'}

features = ['OARSI OST-FL', 'OARSI OST-TL', 'OARSI JSN-L',
            'OARSI OST-FM', 'OARSI OST-TM', 'OARSI JSN-M']


def calc_f1_weighted(y_true, preds, digits=4):
    clf_rep = classification_report(y_true, preds, digits=digits)
    f1_weighted = float(clf_rep.split('\n')[-2].split()[-2])
    return f1_weighted


def calc_f1_macro(y_true, preds, digits=4):
    clf_rep = classification_report(y_true, preds, digits=digits)
    f1_macro = float(clf_rep.split('\n')[-3].split()[-2]) # sklearn
    return f1_macro


def report_scores(gt, predicts_kl, predicts_oarsi, seed, save_dir, digits, n_bootstrap):
    predicts_kl = predicts_kl[:, gt[:, 0].min():(gt[:, 0].max() + 1)].argmax(1)
    if save_dir is not None:
        plot_confusion(gt[:, 0], predicts_kl, os.path.join(save_dir, 'conf_kl.pdf'))
    # we need to make sure that softmax outputs do not produce more than we compare to

    f1_weighted, f1_ci_l, f1_ci_h = bootstrap_ci(partial(calc_f1_weighted, digits=digits),
                                                 gt[:, 0], predicts_kl,
                                                 n_bootstrap, seed=seed)

    f1_macro, f1_m_ci_l, f1_m_ci_h = bootstrap_ci(partial(calc_f1_macro, digits=digits),
                                                 gt[:, 0], predicts_kl,
                                                 n_bootstrap, seed=seed)

    mse, mse_ci_l, mse_ci_h = bootstrap_ci(mean_squared_error,
                                           gt[:, 0], predicts_kl,
                                           n_bootstrap, seed=seed)
    acc, acc_ci_l, acc_ci_h = bootstrap_ci(balanced_accuracy_score,
                                           gt[:, 0], predicts_kl,
                                           n_bootstrap, seed=seed)
    kappa, kappa_ci_l, kappa_ci_h = bootstrap_ci(partial(cohen_kappa_score, weights='quadratic'),
                                                 gt[:, 0], predicts_kl,
                                                 n_bootstrap, seed=seed)
    print(f'=======> KL')
    print(f'{np.round(f1_weighted, digits)} '
          f'({np.round(f1_ci_l, digits)}-{np.round(f1_ci_h, digits)}) & '
          f'{np.round(f1_macro, digits)} '
          f'({np.round(f1_m_ci_l, digits)}-{np.round(f1_m_ci_h, digits)}) & '
          f'{np.round(mse, digits)} '
          f'({np.round(mse_ci_l, digits)}-{np.round(mse_ci_h, digits)}) & '
          f'{np.round(acc, digits)} '
          f'({np.round(acc_ci_l, digits)}-{np.round(acc_ci_h, digits)})  & '
          f'{np.round(kappa, digits)} '
          f'({np.round(kappa_ci_l, digits)}-{np.round(kappa_ci_h, digits)}) \\\\')

    for feature_id, feature_name in enumerate(features):
        feature_pred = predicts_oarsi[:, feature_id, :].argmax(1)
        if save_dir is not None:
            plot_confusion(gt[:, feature_id + 1].astype(int), feature_pred,
                           os.path.join(save_dir, f'conf_{feature_name}.pdf'), font=20)

        f1_weighted, f1_ci_l, f1_ci_h = bootstrap_ci(
            partial(calc_f1_weighted, digits=digits),
            gt[:, feature_id + 1],
            feature_pred,
            n_bootstrap, seed=seed)

        f1_macro, f1_m_ci_l, f1_m_ci_h = bootstrap_ci(partial(calc_f1_macro, digits=digits),
                                                      gt[:, feature_id + 1], feature_pred,
                                                      n_bootstrap, seed=seed)

        mse, mse_ci_l, mse_ci_h = bootstrap_ci(mean_squared_error,
                                               gt[:, feature_id + 1],
                                               feature_pred,
                                               n_bootstrap, seed=seed)

        acc, acc_ci_l, acc_ci_h = bootstrap_ci(balanced_accuracy_score,
                                               gt[:, feature_id + 1],
                                               feature_pred,
                                               n_bootstrap, seed=seed)

        kappa, kappa_ci_l, kappa_ci_h = bootstrap_ci(
            partial(cohen_kappa_score, weights='quadratic'),
            gt[:, feature_id + 1],
            feature_pred,
            n_bootstrap, seed=seed)

        print(f'=======> ' + feature_name)

        print(f'{np.round(f1_weighted, digits)} '
              f'({np.round(f1_ci_l, digits)}-{np.round(f1_ci_h, digits)}) & '
              f'{np.round(f1_macro, digits)} '
              f'({np.round(f1_m_ci_l, digits)}-{np.round(f1_m_ci_h, digits)}) & '
              f'{np.round(mse, digits)} '
              f'({np.round(mse_ci_l, digits)}-{np.round(mse_ci_h, digits)}) & '
              f'{np.round(acc, digits)} '
              f'({np.round(acc_ci_l, digits)}-{np.round(acc_ci_h, digits)})  & '
              f'{np.round(kappa, digits)} '
              f'({np.round(kappa_ci_l, digits)}-{np.round(kappa_ci_h, digits)}) \\\\')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots_dir', default='/mnt/FAST/OARSI_grading_project/KneeOARSIGrading/workdir/snapshots')
    parser.add_argument('--only_first_fu', type=bool, default=False)
    parser.add_argument('--precision', type=int, default=2)
    parser.add_argument('--n_bootstrap', type=int, default=500)
    parser.add_argument('--save_dir', default='/mnt/FAST/OARSI_grading_project/KneeOARSIGrading/workdir/Results/pics/')
    parser.add_argument('--model_list', nargs='+', default=['ens_se_resnet50_se_resnext50_32x4d', ])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
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
                for model in args.model_list:
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
                        gt = data['gt'].astype(int)

                        if len(features)+1 != gt.shape[1]:
                            continue

                        visits = data['visits']
                        if args.only_first_fu:
                            ind_take = np.array(list(map(lambda x: x == '00', visits)))
                        else:
                            ind_take = np.arange(visits.shape[0], dtype=np.int64)
                        predicts_kl = data['predicts_kl']

                        print(f'====> {model} [{snp}]')

                        predicts_kl = predicts_kl[ind_take, :]
                        gt = gt[ind_take, :]
                        predicts_oarsi = data['predicts_oarsi'][ind_take, :, :]
                        report_scores(gt, predicts_kl, predicts_oarsi,
                                      args.seed, args.save_dir, args.precision, args.n_bootstrap)
                        print('#'*80)
                        print('# Evaluation KL-wise')
                        print('#' * 80)
                        for kl_baseline in [0, 2, 3]:
                            print(f'======= KL {kl_baseline} =======')
                            if kl_baseline == 0:
                                ind_take = (gt[:, 0] == 0) | (gt[:, 0] == 1)
                            elif kl_baseline == 3:
                                ind_take = (gt[:, 0] == 3) | (gt[:, 0] == 4)
                            report_scores(gt[ind_take, :], predicts_kl[ind_take], predicts_oarsi[ind_take, :],
                                          args.seed, None, args.precision, args.n_bootstrap)
