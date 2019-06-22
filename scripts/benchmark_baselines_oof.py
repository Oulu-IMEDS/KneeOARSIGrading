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
    parser.add_argument('--phase', choices=['oof', 'test'], default='oof')
    parser.add_argument('--precision', type=int, default=4)
    args = parser.parse_args()

    for metric in ['kappa', 'acc']:
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
                        for snp in glob.glob(os.path.join(args.snapshots_dir, '*', f'{args.phase}_inference',
                                                          'metrics_plain.json')):
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

                            template = '{0}                           '
                            if args.precision == 2:
                                template += '& {1:.2f}         '
                                template += '& {2:.2f}                  '
                                template += '& {3:.2f}         '
                                template += '& {4:.2f}                  '
                                template += '& {5:.2f}         '
                                template += '& {6:.2f}                  '
                                template += '& {7:.2f}                  '
                            elif args.precision == 3:
                                template += '& {1:.3f}         '
                                template += '& {2:.3f}                  '
                                template += '& {3:.3f}         '
                                template += '& {4:.3f}                  '
                                template += '& {5:.3f}         '
                                template += '& {6:.3f}                  '
                                template += '& {7:.3f}                  '
                            elif args.precision == 4:
                                template += '& {1:.4f}         '
                                template += '& {2:.4f}                  '
                                template += '& {3:.4f}         '
                                template += '& {4:.4f}                  '
                                template += '& {5:.4f}         '
                                template += '& {6:.4f}                  '
                                template += '& {7:.4f}                  '

                            template += '\\'
                            template += '\\'
                            mult = 1
                            if metric == 'acc':
                                mult = 100
                            res = template.format(model_dict[oof_res['model']['backbone']],
                                                  np.round(oof_res.get(f'{metric}_kl', -1), args.precision)*mult,
                                                  np.round(oof_res[f'{metric}_osfl'], args.precision)*mult,
                                                  np.round(oof_res[f'{metric}_osfm'], args.precision)*mult,
                                                  np.round(oof_res[f'{metric}_ostl'], args.precision)*mult,
                                                  np.round(oof_res[f'{metric}_ostm'], args.precision)*mult,
                                                  np.round(oof_res[f'{metric}_jsl'], args.precision)*mult,
                                                  np.round(oof_res[f'{metric}_jsm'], args.precision)*mult
                                                  )

                            print(res)
                            print()
