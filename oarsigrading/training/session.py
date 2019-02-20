import operator
import os
import sys
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection  import GroupKFold


from oarsigrading.dataset.utils import build_dataset_meta
from oarsigrading.kvs import GlobalKVS, git_info
from oarsigrading.training.args import parse_args
from oarsigrading.training.dataset import OARSIGradingDataset
from oarsigrading.dataset.utils import make_weights_for_multiclass, WeightedRandomSampler
from oarsigrading.training.transforms import init_transforms
from oarsigrading.training.utils import net_core
DEBUG = sys.gettrace() is not None


def init_session():
    if not torch.cuda.is_available():
        raise EnvironmentError('The code must be run on GPU.')

    kvs = GlobalKVS()

    # Getting the arguments
    args = parse_args()
    # Initializing the seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Creating the snapshot
    snapshot_name = time.strftime('%Y_%m_%d_%H_%M')
    os.makedirs(os.path.join(args.snapshots, snapshot_name), exist_ok=True)

    res = git_info()
    if res is not None:
        kvs.update('git branch name', res[0])
        kvs.update('git commit id', res[1])
    else:
        kvs.update('git branch name', None)
        kvs.update('git commit id', None)

    kvs.update('pytorch_version', torch.__version__)

    if torch.cuda.is_available():
        kvs.update('cuda', torch.version.cuda)
        kvs.update('gpus', torch.cuda.device_count())
    else:
        kvs.update('cuda', None)
        kvs.update('gpus', None)

    kvs.update('snapshot_name', snapshot_name)
    kvs.update('args', args)
    kvs.save_pkl(os.path.join(args.snapshots, snapshot_name, 'session.pkl'))

    return args, snapshot_name


def init_metadata():
    kvs = GlobalKVS()
    if not os.path.isfile(os.path.join(kvs['args'].snapshots, 'oai_meta.pkl')):
        print('==> Cached metadata is not found. Generating...')
        oai_meta, most_meta = build_dataset_meta(kvs['args'])
        oai_meta.to_pickle(os.path.join(kvs['args'].snapshots, 'oai_meta.pkl'), compression='infer', protocol=4)
        most_meta.to_pickle(os.path.join(kvs['args'].snapshots, 'most_meta.pkl'), compression='infer', protocol=4)
    else:
        print('==> Loading cached metadata...')
        oai_meta = pd.read_pickle(os.path.join(kvs['args'].snapshots, 'oai_meta.pkl'))

        most_meta = pd.read_pickle(os.path.join(kvs['args'].snapshots, 'most_meta.pkl'))

    most_meta = most_meta[(most_meta.XRKL >= 0) & (most_meta.XRKL <= 4)]
    oai_meta = oai_meta[(oai_meta.XRKL >= 0) & (oai_meta.XRKL <= 4)]

    print(colored('==> ', 'green') + 'Images in OAI:', oai_meta.shape[0])
    print(colored('==> ', 'green') + 'Images in MOST:', most_meta.shape[0])

    kvs.update('most_meta', most_meta)
    kvs.update('oai_meta', oai_meta)

    gkf = GroupKFold(kvs['args'].n_folds)
    cv_split = [x for x in gkf.split(kvs[kvs["args"].train_set + '_meta'],
                                     groups=kvs[kvs["args"].train_set + '_meta']['ID'].values)]

    kvs.update('cv_split_all_folds', cv_split)
    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))


def init_data_processing():
    kvs = GlobalKVS()

    train_trf, val_trf = init_transforms(None, None)

    dataset = OARSIGradingDataset(kvs[f'{kvs["args"].train_set}_meta'], train_trf)

    mean_vector, std_vector = init_mean_std(snapshots_dir=kvs['args'].snapshots,
                                            dataset=dataset, batch_size=kvs['args'].bs,
                                            n_threads=kvs['args'].n_threads)

    print(colored('====> ', 'red') + 'Mean:', mean_vector)
    print(colored('====> ', 'red') + 'Std:', std_vector)

    kvs.update('mean_vector', mean_vector)
    kvs.update('std_vector', std_vector)

    train_trf, val_trf = init_transforms(mean_vector, std_vector)

    kvs.update('train_trf', train_trf)
    kvs.update('val_trf', val_trf)
    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))


def init_mean_std(snapshots_dir, dataset, batch_size, n_threads):
    kvs = GlobalKVS()
    if os.path.isfile(os.path.join(snapshots_dir, f'mean_std_{kvs["args"].train_set}.npy')):
        tmp = np.load(os.path.join(snapshots_dir, f'mean_std_{kvs["args"].train_set}.npy'))
        mean_vector, std_vector = tmp
    else:
        tmp_loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_threads)
        mean_vector = None
        std_vector = None
        print(colored('==> ', 'green') + 'Calculating mean and std')
        for batch in tqdm(tmp_loader, total=len(tmp_loader)):
            imgs = batch['img']
            if mean_vector is None:
                mean_vector = np.zeros(imgs.size(1))
                std_vector = np.zeros(imgs.size(1))
            for j in range(mean_vector.shape[0]):
                mean_vector[j] += imgs[:, j, :, :].mean()
                std_vector[j] += imgs[:, j, :, :].std()

        mean_vector /= len(tmp_loader)
        std_vector /= len(tmp_loader)
        np.save(os.path.join(snapshots_dir, f'mean_std_{kvs["args"].train_set}.npy'),
                [mean_vector.astype(np.float32), std_vector.astype(np.float32)])

    return mean_vector, std_vector


def init_datasets(x_train, x_val):
    kvs = GlobalKVS()
    train_dataset = OARSIGradingDataset(x_train, kvs['train_trf'])
    val_dataset = OARSIGradingDataset(x_val, kvs['val_trf'])

    return train_dataset, val_dataset


def init_loaders(x_train, x_val):
    kvs = GlobalKVS()
    train_dataset, val_dataset = init_datasets(x_train, x_val)

    if kvs['args'].weighted_sampling:
        print(colored('====> ', 'red') + 'Using weighted sampling')
        _, weights = make_weights_for_multiclass(x_train.XRKL.values.astype(int))
        sampler = WeightedRandomSampler(weights, x_train.shape[0], True)

        train_loader = DataLoader(train_dataset, batch_size=kvs['args'].bs,
                                  num_workers=kvs['args'].n_threads,
                                  drop_last=True, sampler=sampler)

    else:
        train_loader = DataLoader(train_dataset, batch_size=kvs['args'].bs,
                                  num_workers=kvs['args'].n_threads,
                                  drop_last=True, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=kvs['args'].val_bs,
                            num_workers=kvs['args'].n_threads)

    return train_loader, val_loader


def init_folds():
    kvs = GlobalKVS()
    writers = {}
    cv_split_train = {}
    for fold_id, split in enumerate(kvs['cv_split_all_folds']):
        if kvs['args'].fold != -1 and fold_id != kvs['args'].fold:
            continue
        kvs.update(f'losses_fold_[{fold_id}]', None, list)
        kvs.update(f'val_metrics_fold_[{fold_id}]', None, list)
        cv_split_train[fold_id] = split
        writers[fold_id] = SummaryWriter(os.path.join(kvs['args'].snapshots,
                                                      kvs['snapshot_name'],
                                                      'logs',
                                                      'fold_{}'.format(fold_id), kvs['snapshot_name']))

    kvs.update('cv_split_train', cv_split_train)
    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))
    return writers


def save_checkpoint(model, optimizer):
    kvs = GlobalKVS()
    fold_id = kvs['cur_fold']
    epoch = kvs['cur_epoch']
    val_metric = kvs[f'val_metrics_fold_[{fold_id}]'][-1][0][kvs['args'].snapshot_on]
    comparator = getattr(operator, kvs['args'].snapshot_comparator)
    cur_snapshot_name = os.path.join(kvs['args'].snapshots, kvs['snapshot_name'],
                                     f'fold_{fold_id}_epoch_{epoch+1}.pth')

    if kvs['prev_model'] is None:
        print(colored('====> ', 'red') + 'Snapshot was saved to', cur_snapshot_name)
        torch.save({'epoch': epoch, 'net': net_core(model).state_dict(),
                    'optim': optimizer.state_dict()}, cur_snapshot_name)

        kvs.update('prev_model', cur_snapshot_name)
        kvs.update('best_val_metric', val_metric)

    else:
        if comparator(val_metric, kvs['best_val_metric']):
            print(colored('====> ', 'red') + 'Snapshot was saved to', cur_snapshot_name)
            if not kvs['args'].keep_snapshots:
                os.remove(kvs['prev_model'])

            torch.save({'epoch': epoch, 'net': net_core(model).state_dict(),
                        'optim': optimizer.state_dict()}, cur_snapshot_name)
            kvs.update('prev_model', cur_snapshot_name)
            kvs.update('best_val_metric', val_metric)

    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))
