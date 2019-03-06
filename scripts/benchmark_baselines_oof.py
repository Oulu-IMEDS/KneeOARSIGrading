import glob
import json
import numpy as np
import os
import argparse

model_dict = {'resnet18': 'Resnet-18',
             'resnet34': 'Resnet-34',
             'resnet50': 'Resnet-50',
             'se_resnet50': 'SE-Resnet-50',
             'se_resnext50_32x4d': 'SE-ResNext50-32x4d'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots_dir', default='/media/lext/FAST/OARSI_grading_project/workdir/'
                                                   'oarsi_grades_snapshots_gwap/')
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
                    for snp in glob.glob(os.path.join(args.snapshots_dir, '*', 'oof_inference/metrics_plain.json')):
                        with open(snp, 'r') as f:
                            oof_res = json.load(f)

                        if oof_res['model']['backbone'] != model:
                            continue
                        if oof_res['model']['weighted_sampling'] != weighted:
                            continue
                        if oof_res['model']['gwap'] != gwap:
                            continue
                        if oof_res['model']['gwap_hidden'] != gwap_hidden:
                            continue

                        template = '& {0}                           '
                        template += '& {1:.4f}                   '
                        template += '& {1:.4f}         '
                        template += '& {2:.4f}                  '
                        template += '& {3:.4f}         '
                        template += '& {4:.4f}                  '
                        template += '& {5:.4f}         '
                        template += '& {6:.4f}                  '
                        template += '\\'

                        template += '\\'

                        res = template.format(model_dict[oof_res['model']['backbone']],
                                              np.round(oof_res['kappa_kl'], 4),
                                              np.round(oof_res['kappa_osfl'], 4),
                                              np.round(oof_res['kappa_osfm'], 4),
                                              np.round(oof_res['kappa_ostl'], 4),
                                              np.round(oof_res['kappa_ostm'], 4),
                                              np.round(oof_res['kappa_jsl'], 4),
                                              np.round(oof_res['kappa_jsm'], 4)
                                              )

                        print(res)
                        print()
