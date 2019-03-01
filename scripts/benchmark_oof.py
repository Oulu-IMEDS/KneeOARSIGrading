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

    for model in ['resnet18', 'resnet34', 'resnet50', 'se_resnet50', 'se_resnext50_32x4d']:
        data_dir = os.path.join(args.snapshots_dir, '*', 'oof_inference/metrics.json')
        for snp in glob.glob(os.path.join(args.snapshots_dir, '*', 'oof_inference/metrics_plain.json')):
            with open(snp, 'r') as f:
                oof_res = json.load(f)

            if oof_res['model']['backbone'] != model:
                continue

            if not oof_res['model']['weighted_sampling']:
                continue

            print(oof_res['model']['backbone'],
                  'GWAP' if oof_res['model']['gwap'] else 'AVG_pool',
                  'GWAP_hidden' if oof_res['model']['gwap_hidden'] else '-',
                  'WS' if oof_res['model']['weighted_sampling'] else 'NWS')

            template = '& {0}                           '
            template += '& {1:.2f}                   '
            template += '& {1:.2f}         '
            template += '& {2:.2f}                  '
            template += '& {3:.2f}         '
            template += '& {4:.2f}                  '
            template += '& {5:.2f}         '
            template += '& {6:.2f}                  '
            template += '\\'

            template += '\\'

            res = template.format(model_dict[oof_res['model']['backbone']],
                                  np.round(oof_res['kappa_kl'], 2),
                                  np.round(oof_res['kappa_osfl'], 2),
                                  np.round(oof_res['kappa_osfm'], 2),
                                  np.round(oof_res['kappa_ostl'], 2),
                                  np.round(oof_res['kappa_ostm'], 2),
                                  np.round(oof_res['kappa_jsl'], 2),
                                  np.round(oof_res['kappa_jsm'], 2)
                                  )

            print(res)
            print()
