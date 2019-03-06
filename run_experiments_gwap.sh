#!/usr/bin/env bash

SNAPSHOTS_DIR=/media/lext/FAST/OARSI_grading_project/workdir/oarsi_grades_snapshots_gwap
DATA_DIR=/media/lext/FAST/OARSI_grading_project/Data/datasets/
META_DIR=/media/lext/FAST/OARSI_grading_project/
SNP_PREF=2019_02

cd scripts

# GWAP

#python train.py --backbone_depth 18 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR} --use_gwap True

#python train.py --backbone_depth 34 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR} --use_gwap True

#python train.py --backbone_depth 50 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR} --use_gwap True

#python train.py --backbone_depth 50 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR} --use_gwap True --se True

#python train.py --backbone_depth 50 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR} --use_gwap True --se True --dw True

# GWAP hidden used

#python train.py --backbone_depth 18 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR} --use_gwap True --use_gwap_hidden True

#python train.py --backbone_depth 34 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR} --use_gwap True --use_gwap_hidden True

#python train.py --backbone_depth 50 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR} --use_gwap True --use_gwap_hidden True

#python train.py --backbone_depth 50 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR} --use_gwap True --se True --use_gwap_hidden True

#python train.py --backbone_depth 50 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR} --use_gwap True --se True --dw True --use_gwap_hidden True



for SNP_NAME in $(ls ${SNAPSHOTS_DIR} | grep ${SNP_PREF});
do
    python oof_inference.py --snapshots ${SNAPSHOTS_DIR} \
        --dataset_root ${DATA_DIR} \
        --meta_root ${META_DIR}\
        --snapshot ${SNP_NAME}
done

