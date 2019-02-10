import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--fold', type=int, default=-1, help='Fold to train. -1 means train all folds in a row')
    parser.add_argument('--dataset_root', type=str, default='/media/lext/FAST/OARSI_grading_project/Data/datasets/')
    parser.add_argument('--meta_root', type=str, default='/media/lext/FAST/OARSI_grading_project/')
    parser.add_argument('--train_set', type=str, default='oai', choices=['most', 'oai'],
                        help='Dataset to be used for testing.')

    parser.add_argument('--backbone_width', type=int, default=50, help='Width of SE-Resnet')
    parser.add_argument('--imsize', type=int, default=700)
    parser.add_argument('--crop_size', type=int, default=650)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument('--snapshot_on', type=str, choices=['val_loss', ], default='val_loss')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--val_bs', type=int, default=64, help='Batch size')
    parser.add_argument('--unfreeze_epoch', type=int, default=1,
                        help='Epoch at which to unfreeze the layers of the backbone')

    parser.add_argument('--snapshots', default='/media/lext/FAST/OARSI_grading_project/workdir/oarsi_grades_snapshots/',
                        help='Folder for saving snapshots')
    parser.add_argument('--n_threads', default=12, type=int, help='Number of parallel threads for Data Loader')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    args = parser.parse_args()

    return args
