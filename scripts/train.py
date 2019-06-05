import cv2
import sys
from termcolor import colored

from oarsigrading.kvs import GlobalKVS
from oarsigrading.training import session
from oarsigrading.training import utils
from oarsigrading.evaluation import metrics

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

DEBUG = sys.gettrace() is not None

if __name__ == "__main__":
    kvs = GlobalKVS()
    session.init_session()
    session.init_metadata()
    writers = session.init_folds()
    session.init_data_processing()

    for fold_id in kvs['cv_split_train']:
        if kvs['args'].fold != -1 and fold_id != kvs['args'].fold:
            continue

        kvs.update('cur_fold', fold_id)
        kvs.update('prev_model', None)
        print(colored('====> ', 'blue') + f'Training fold {fold_id}....')

        train_index, val_index = kvs['cv_split_train'][fold_id]
        train_loader, val_loader = session.init_loaders(kvs[f'{kvs["args"].train_set}_meta'].iloc[train_index],
                                                        kvs[f'{kvs["args"].train_set}_meta'].iloc[val_index])

        net, criterion = utils.init_model()

        if kvs['args'].pretrained:
            net.train(False)
            utils.net_core(net).classifier.train(True)
            optimizer = utils.init_optimizer(utils.layer_params(net, 'classifier'))
        else:
            print(colored('====> ', 'red') + 'The model will be trained from scratch!')
            net.train(True)
            optimizer = utils.init_optimizer(net.parameters())

        scheduler = utils.init_scheduler(optimizer, 0)
        for epoch in range(kvs['args'].n_epochs):
            scheduler.step()
            kvs.update('cur_epoch', epoch)
            if kvs['args'].pretrained:
                if epoch == kvs['args'].unfreeze_epoch:
                    print(colored('====> ', 'red') + 'Unfreezing the layers!')
                    # Making the whole model trainable
                    net.train(True)
                    optimizer.add_param_group({'params': utils.layer_params(net, 'encoder')})
                    scheduler = utils.init_scheduler(optimizer, epoch)

            print(colored('====> ', 'green') + 'Snapshot::', kvs['snapshot_name'])
            print(colored('====> ', 'red') + 'LR:', scheduler.get_lr())

            train_loss = utils.epoch_pass(net, train_loader, criterion, optimizer, writers[fold_id])
            val_out = utils.epoch_pass(net, val_loader, criterion, None, None)
            val_loss, val_ids, gt, preds = val_out
            metrics.log_metrics(writers[fold_id], train_loss, val_loss, gt, preds)
            session.save_checkpoint(net, optimizer)
