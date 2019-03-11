import glob
import json
import numpy as np
import os
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots_dir', default='/media/lext/FAST/OARSI_grading_project/workdir/'
                                                   'oarsi_grades_snapshots_weighing_exp/')
    args = parser.parse_args()

    most_meta = pd.read_pickle(os.path.join(args.snapshots_dir, 'most_meta.pkl'))
    oai_meta = pd.read_pickle(os.path.join(args.snapshots_dir, 'oai_meta.pkl'))

    most_meta = most_meta[(most_meta.XRKL >= 0) & (most_meta.XRKL <= 4)]
    oai_meta = oai_meta[(oai_meta.XRKL >= 0) & (oai_meta.XRKL <= 4)]

    # Train
    ## KL
    counts, _ = np.histogram(oai_meta.XRKL.values, bins=np.arange(6).astype(int))
    res = '{} & {} & {} & {} & {}'.format(counts[0], counts[1], counts[2], counts[3], counts[4])

    ## Lateral
    counts, _ = np.histogram(oai_meta.XROSFL.values, bins=np.arange(5).astype(int))
    res = '& {} & {} & {} & {}'.format(counts[0], counts[1], counts[2], counts[3])

    counts, _ = np.histogram(oai_meta.XROSTL.values, bins=np.arange(5).astype(int))
    res += '& {} & {} & {} & {}'.format(counts[0], counts[1], counts[2], counts[3])

    counts, _ = np.histogram(oai_meta.XRJSL.values, bins=np.arange(5).astype(int))
    res += '& {} & {} & {} & {}'.format(counts[0], counts[1], counts[2], counts[3])
    res += '\\\\'
    print(res)
    ## Medial
    counts, _ = np.histogram(oai_meta.XROSFM.values, bins=np.arange(5).astype(int))
    res = '& {} & {} & {} & {}'.format(counts[0], counts[1], counts[2], counts[3])

    counts, _ = np.histogram(oai_meta.XROSTM.values, bins=np.arange(5).astype(int))
    res += '& {} & {} & {} & {}'.format(counts[0], counts[1], counts[2], counts[3])

    counts, _ = np.histogram(oai_meta.XRJSM.values, bins=np.arange(5).astype(int))
    res += '& {} & {} & {} & {}'.format(counts[0], counts[1], counts[2], counts[3])
    res += '\\\\'
    print(res)

    # Test
    ## KL
    counts, _ = np.histogram(most_meta.XRKL.values, bins=np.arange(6).astype(int))
    res = '{} & {} & {} & {} & {}'.format(counts[0], counts[1], counts[2], counts[3], counts[4])

    ## Lateral
    counts, _ = np.histogram(most_meta.XROSFL.values, bins=np.arange(5).astype(int))
    res = '& {} & {} & {} & {}'.format(counts[0], counts[1], counts[2], counts[3])

    counts, _ = np.histogram(most_meta.XROSTL.values, bins=np.arange(5).astype(int))
    res += '& {} & {} & {} & {}'.format(counts[0], counts[1], counts[2], counts[3])

    counts, _ = np.histogram(most_meta.XRJSL.values, bins=np.arange(5).astype(int))
    res += '& {} & {} & {} & {}'.format(counts[0], counts[1], counts[2], counts[3])
    res += '\\\\'
    print(res)
    ## Medial
    counts, _ = np.histogram(most_meta.XROSFM.values, bins=np.arange(5).astype(int))
    res = '& {} & {} & {} & {}'.format(counts[0], counts[1], counts[2], counts[3])

    counts, _ = np.histogram(most_meta.XROSTM.values, bins=np.arange(5).astype(int))
    res += '& {} & {} & {} & {}'.format(counts[0], counts[1], counts[2], counts[3])

    counts, _ = np.histogram(most_meta.XRJSM.values, bins=np.arange(5).astype(int))
    res += '& {} & {} & {} & {}'.format(counts[0], counts[1], counts[2], counts[3])
    res += '\\\\'
    print(res)


