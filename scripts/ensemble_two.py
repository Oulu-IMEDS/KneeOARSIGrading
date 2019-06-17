import numpy as np
import argparse
import pickle
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--snapshot1', default='')
    parser.add_argument('--snapshot2', default='')
    parser.add_argument('--tta1', type=bool, default=False)
    parser.add_argument('--tta2', type=bool, default=False)

    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot1, 'session.pkl'), 'rb') as f:
        session_1_backup = pickle.load(f)

    with open(os.path.join(args.snapshots_root, args.snapshot1, 'session.pkl'), 'rb') as f:
        session_2_backup = pickle.load(f)

    snapshot_1 = np.load(os.path.join(args.snapshots_root, args.snapshot1, 'oof_inference',
                                      f'results_{"TTA" if args.tta1 else "plain"}.npz'))

    snapshot_2 = np.load(os.path.join(args.snapshots_root, args.snapshot2, 'oof_inference',
                                      f'results_{"TTA" if args.tta2 else "plain"}.npz'))




