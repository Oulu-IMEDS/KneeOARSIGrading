import os
from oarsigrading.dataset.metadata.constants import follow_up_dict_oai
from oarsigrading.dataset.metadata.utils import read_sas7bdat
import pandas as pd


def get_oai_meta(meta_path):
    oai_datasets = []
    for metadata in os.listdir(meta_path):
        visit = f"V{metadata.split('_')[-1][2:4]}"
        print('==> Processing', os.path.join(meta_path, f'{metadata}'))
        oai_meta = read_sas7bdat(os.path.join(meta_path, f'{metadata}'))

        # These can be used in the future if we want.
        feats_full = ['ID', # Patient ID
                 'SIDE',            # Side - left or right knee
                 'VISIT',           # Follow-up visit
                 f'{visit}XRKL',  # KL grade
                 f'{visit}XRSCFL',  # sclerosis (OARSI grades 0- 3) femoral lateral compartment
                 f'{visit}XRSCFM',  # sclerosis (OARSI grades 0- 3) femoral medial compartment
                 f'{visit}XRSCTL',  # sclerosis (OARSI grades 0- 3) tibia lateral compartment
                 f'{visit}XRSCTM',  # sclerosis (OARSI grades 0- 3) tibia medial compartment
                 f'{visit}XROSTL',  # osteophytes (OARSI grades 0- 3) tibia lateral compartment
                 f'{visit}XROSFL',  # osteophytes (OARSI grades 0- 3) femur lateral compartment
                 f'{visit}XROSTM',  # osteophytes (OARSI grades 0- 3) tibia medial compartment
                 f'{visit}XROSFM',  # osteophytes (OARSI grades 0- 3) femur medial compartment
                 f'{visit}XRJSL',   # joint space narrowing (OARSI grades 0- 3) lateral compartment
                 f'{visit}XRJSM',   # joint space narrowing (OARSI grades 0- 3) medial compartment
                 f'{visit}XRATTL',  # attrition (OARSI grades 0- 3) tibia lateral compartment
                 f'{visit}XRATTM',  # attrition (OARSI grades 0- 3) tibia medial compartment
                     ]

        oai_meta['VISIT'] = follow_up_dict_oai[visit]

        try:
            oai_meta = oai_meta[~oai_meta[feats_full].isnull().any(1)][feats_full]
        except:
            continue
        oai_meta[list(map(lambda x: x.split(visit)[-1], feats_full[3:]))] = oai_meta[feats_full[3:]]
        if oai_meta.shape[0] > 0:
            oai_meta.drop(feats_full[3:], axis=1, inplace=True)
            oai_datasets.append(oai_meta)

    return pd.concat(oai_datasets)

