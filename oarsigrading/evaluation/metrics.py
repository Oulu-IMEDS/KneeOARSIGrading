from sklearn.metrics import cohen_kappa_score
from termcolor import colored
import os
from oarsigrading.kvs import GlobalKVS


def compute_metrics(val_pred, val_gt, no_kl=False):
    kappas = []
    for task_id in range(val_gt.shape[1]):
        kappas.append(cohen_kappa_score(val_gt[:, task_id], val_pred[:, task_id], weights='quadratic'))

    if no_kl:
        kappa_res = {'kappa_ostl': kappas[0],
                     'kappa_osfl': kappas[1],
                     'kappa_jsl': kappas[2],
                     'kappa_ostm': kappas[3],
                     'kappa_osfm': kappas[4],
                     'kappa_jsm': kappas[5]
                     }
    else:
        kappa_res = {'kappa_kl': kappas[0],
                     'kappa_ostl': kappas[1],
                     'kappa_osfl': kappas[2],
                     'kappa_jsl': kappas[3],
                     'kappa_ostm': kappas[4],
                     'kappa_osfm': kappas[5],
                     'kappa_jsm': kappas[6]
                     }

    print(colored('==> ', 'green') + f' Kappas:')
    for k in kappa_res:
        print(colored('==> ', 'red') + f'{k} : {kappa_res[k]:.4f}')

    return kappa_res


def log_metrics(boardlogger, train_loss, val_loss, val_pred, val_gt):
    kvs = GlobalKVS()
    res = {
        'epoch': kvs['cur_epoch'],
        'val_loss': val_loss
    }
    print(colored('==> ', 'green') + f'Train loss: {train_loss:.4f} / Val loss: {val_loss:.4f}')

    res.update(compute_metrics(val_pred, val_gt, no_kl=kvs['args'].no_kl))

    boardlogger.add_scalars('Losses', {'train': train_loss, 'val': val_loss}, kvs['cur_epoch'])
    boardlogger.add_scalars('Metrics', {metric: res[metric] for metric in res if metric.startswith('kappa')},
                            kvs['cur_epoch'])

    kvs.update(f'losses_fold_[{kvs["cur_fold"]}]', {'epoch': kvs['cur_epoch'],
                                                    'train_loss': train_loss,
                                                    'val_loss': val_loss})

    kvs.update(f'val_metrics_fold_[{kvs["cur_fold"]}]', res)

    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))