import os
from oarsigrading.metadata.constants import follow_up_dict_most
from oarsigrading.metadata.utils import read_sas7bdat
import pandas as pd


def get_most_meta(meta_path):
    print('==> Processing', os.path.join(meta_path, 'mostv01235xray.sas7bdat'))
    most_meta = read_sas7bdat(os.path.join(meta_path, 'mostv01235xray.sas7bdat'))

    most_names_list = pd.read_csv(os.path.join(meta_path, 'MOST_names.csv'), header=None)[0].values.tolist()
    xray_types = pd.DataFrame(
        list(map(lambda x: (x.split('/')[0][:-5], follow_up_dict_most[int(x.split('/')[1][1])], x.split('/')[-2]),
                 most_names_list)), columns=['ID', 'VISIT', 'TYPE'])

    most_meta_all = []
    for visit_id in [0, 1, 2, 3, 5]:
        for leg in ['L', 'R']:
            features = ['MOSTID', ]
            for compartment in ['L', 'M']:
                for bone in ['F', 'T']:
                    features.append(f"V{visit_id}X{leg}OS{bone}{compartment}"),
                features.append(f"V{visit_id}X{leg}JS{compartment}")
            tmp = most_meta.copy()[features]
            trunc_feature_names = list(map(lambda x: 'XR' + x[4:], features[1:]))
            tmp[trunc_feature_names] = tmp[features[1:]]
            tmp.drop(features[1:], axis=1, inplace=True)
            tmp['SIDE'] = int(1 if leg == 'L' else 2)
            tmp = tmp[~tmp.isnull().any(1)]
            tmp['VISIT'] = follow_up_dict_most[visit_id]
            tmp['ID'] = tmp['MOSTID'].copy()
            tmp.drop('MOSTID', axis=1, inplace=True)
            most_meta_all.append(tmp)

    most_meta = pd.concat(most_meta_all)
    most_meta = most_meta[(most_meta[trunc_feature_names] <= 3).all(1)]
    most_meta = pd.merge(xray_types, most_meta)
    most_meta = most_meta[most_meta.TYPE == 'PA10']
    most_meta.drop('TYPE', axis=1, inplace=True)
    return most_meta

