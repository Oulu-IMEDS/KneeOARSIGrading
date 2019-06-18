from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
from termcolor import colored
import os
from oarsigrading.kvs import GlobalKVS


def compute_metrics(val_pred, val_gt, no_kl=False):
    kappas = []
    acc = []
    for task_id in range(val_gt.shape[1]):
        kappas.append(cohen_kappa_score(val_gt[:, task_id], val_pred[:, task_id], weights='quadratic'))
        acc.append(balanced_accuracy_score(val_gt[:, task_id], val_pred[:, task_id]))

    res = {}
    ind = 0
    for feature in ['kl', 'ostl', 'osfl', 'jsl', 'ostm', 'osfm', 'jsm']:
        if no_kl and feature == 'kl':
            continue
        res[f"kappa_{feature}"] = kappas[ind]
        res[f"acc_{feature}"] = acc[ind]
        ind += 1

    print(colored('==> ', 'green') + f' Kappas:')
    for k in res:
        if 'kappa' in k:
            print(colored('==> ', 'red') + f'{k} : {res[k]:.4f}')
    print(colored('==> ', 'green') + f' Balanced Accuracy:')
    for k in res:
        if 'acc' in k:
            print(colored('==> ', 'red') + f'{k} : {res[k]:.4f}')

    return res


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