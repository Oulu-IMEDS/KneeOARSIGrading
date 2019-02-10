import cv2
import sys
from termcolor import colored
from oarsigrading.kvs import GlobalKVS
from oarsigrading.training import session


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
