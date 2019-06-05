import glob
import json
import numpy as np
import os
import pandas as pd
import argparse


def plot_table_data(dataset, data_name):
    print('\\hline')
    matr = []
    for feature in ['XRKL', 'XROSF', 'XROST', 'XRJS']:
        if feature == 'XRKL':
            vals = dataset['XRKL'].values
            counts, _ = np.histogram(vals, bins=np.arange(6).astype(int))
            matr.append(counts)
        else:
            for comp in ['L', 'M']:
                vals = dataset[feature + comp].values
                counts, _ = np.histogram(vals, bins=np.arange(6).astype(int))
                matr.append(counts)

    print(f'\\multirow{{5}}{{*}}{{{data_name}}}  ', end='')
    print(f'& \\multirow{{5}}{{*}}{{{dataset.shape[0]}}}  ', end='')
    for grade_n, row in enumerate(np.vstack(matr).T):
        if grade_n > 0:
            print('& ', end='')
        print('& ' + str(grade_n) + ' & '+ ' & '.join([f'{el if el != 0 else "-"}' for el in row]) + '\\\\')
    print('\\hline')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots_dir', default='/media/lext/FAST/OARSI_grading_project/workdir/'
                                                   'oarsi_grades_snapshots_weighing_exp/')
    args = parser.parse_args()

    most_meta = pd.read_pickle(os.path.join(args.snapshots_dir, 'most_meta.pkl'))
    oai_meta = pd.read_pickle(os.path.join(args.snapshots_dir, 'oai_meta.pkl'))

    most_meta = most_meta[(most_meta.XRKL >= 0) & (most_meta.XRKL <= 4)]
    oai_meta = oai_meta[(oai_meta.XRKL >= 0) & (oai_meta.XRKL <= 4)]
    plot_table_data(oai_meta, 'OAI')
    plot_table_data(most_meta, 'MOST')



