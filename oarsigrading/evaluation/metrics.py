from sklearn.metrics import cohen_kappa_score
from termcolor import colored
import os
from oarsigrading.kvs import GlobalKVS


def log_metrics(boardlogger, train_loss, val_loss, val_pred, val_gt):
    kvs = GlobalKVS()
    res = {
        'epoch': kvs['cur_epoch'],
        'val_loss': val_loss
    }
    kappas = [cohen_kappa_score(gt, pred, weights='quadratic') for (gt, pred) in zip(val_gt, val_pred)]
    res.update({'kappa_ostl': kappas[0],
                'kappa_osfl': kappas[1],
                'kappa_jsl': kappas[2],
                'kappa_ostm': kappas[3],
                'kappa_osfm': kappas[4],
                'kappa_jsm': kappas[5]}
               )

    boardlogger.add_scalars('Losses', {'train': train_loss, 'val': val_loss}, kvs['cur_epoch'])
    boardlogger.add_scalars('Metrics', {metric: res[metric] for metric in res if metric.startswith('kappa')},
                            kvs['cur_epoch'])

    kvs.update(f'losses_fold_[{kvs["cur_fold"]}]', {'epoch': kvs['cur_epoch'],
                                                    'train_loss': train_loss,
                                                    'val_loss': val_loss})

    kvs.update(f'val_metrics_fold_[{kvs["cur_fold"]}]', res)

    print(colored('==> ', 'green') + f'Train loss: {train_loss:.4f} / Val loss: {val_loss:.4f}')

    print(colored('==> ', 'green') + f' Kappas: {kappas}')

    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))