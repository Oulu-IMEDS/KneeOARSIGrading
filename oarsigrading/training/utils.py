
import gc
from typing import Tuple, List

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.nn import DataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from oarsigrading.kvs import GlobalKVS
from oarsigrading.training.model import MultiTaskClassificationLoss, OARSIGradingNet


def net_core(net: nn.Module) -> nn.Module:
    if isinstance(net, DataParallel):
        return net.module
    else:
        return net


def layer_params(net: nn.Module, layer_name: str):
    return getattr(net_core(net), layer_name).parameters()


def init_loss() -> nn.Module:
    return MultiTaskClassificationLoss()


def init_model() -> Tuple[nn.Module, nn.Module]:
    kvs = GlobalKVS()

    net = OARSIGradingNet(bb_width=kvs['args'].backbone_width, dropout=kvs['args'].dropout_rate,
                          cls_bnorm=kvs['args'].use_bnorm)

    if kvs['gpus'] > 1:
        net = nn.DataParallel(net).to('cuda')

    return net.to('cuda'), init_loss().to('cuda')


def init_optimizer(params) -> Optimizer:
    kvs = GlobalKVS()
    if kvs['args'].optimizer == 'adam':
        return optim.Adam(params, lr=kvs['args'].lr, weight_decay=kvs['args'].wd)
    elif kvs['args'].optimizer == 'sgd':
        return optim.SGD(params, lr=kvs['args'].lr,
                         weight_decay=kvs['args'].wd,
                         momentum=kvs['args'].momentum,
                         nesterov=kvs['args'].nesterov)
    else:
        raise NotImplementedError


def epoch_pass(net: nn.Module, loader: DataLoader, criterion: nn.Module,
               optimizer: Optimizer or None,
               writer: SummaryWriter or None = None) -> float or Tuple[float, List[str], np.ndarray, np.ndarray]:

    kvs = GlobalKVS()
    if optimizer is not None:
        net.train(True)
    else:
        net.train(False)

    running_loss = 0.0
    n_batches = len(loader)
    pbar = tqdm(total=len(loader))
    epoch = kvs['cur_epoch']
    max_epoch = kvs['args'].n_epochs
    fold_id = kvs['cur_fold']

    device = next(net.parameters()).device
    predicts = []
    fnames = []
    gt = []
    with torch.set_grad_enabled(optimizer is not None):
        for i, batch in enumerate(loader):
            if optimizer is not None:
                optimizer.zero_grad()

            # forward + backward + optimize
            labels = batch['target'].squeeze().to(device)
            inputs = batch['img'].squeeze().to(device)

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            if optimizer is not None:
                loss.backward()
                optimizer.step()
                pbar.set_description(f'[{fold_id}] Train:: [{epoch} / {max_epoch}]:: '
                                     f'{running_loss / (i + 1):.3f} | {loss.item():.3f}')
            else:
                """
                for task_id, o in enumerate(outputs):
                    if len(predicts) != len(outputs):
                        predicts.append(o.cpu().numpy().argmax(1).tolist())
                    else:
                        predicts[task_id].extend(o.cpu().numpy().argmax(1).tolist())
                """
                fnames.extend(batch['ID'])

                gt.extend(batch['target'].to('cpu').numpy().squeeze()[:, 0].tolist())
                predicts.extend(outputs.to('cpu').numpy().argmax(1).squeeze().tolist())

                pbar.set_description(f'[{fold_id}] Validating [{epoch} / {max_epoch}]:')
            if writer is not None and optimizer is not None:
                writer.add_scalar('train_logs/loss', loss.item(), kvs['cur_epoch'] * len(loader) + i)
            running_loss += loss.item()
            pbar.update()

            gc.collect()
    gc.collect()
    pbar.close()
    if optimizer is not None:
        return running_loss / n_batches

    return running_loss / n_batches, fnames, np.array(gt), np.array(predicts)
    # np.vstack(gt).squeeze(), np.vstack(predicts).T.copy()


def init_scheduler(optimizer: Optimizer, epoch_start: int) -> MultiStepLR:
    kvs = GlobalKVS()
    scheduler = MultiStepLR(optimizer,
                            milestones=list(map(lambda x: x - epoch_start,
                                                kvs['args'].lr_drop)), gamma=kvs['args'].lr_drop_gamma)

    return scheduler

