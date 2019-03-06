#!/usr/bin/env bash

SNAPSHOTS_DIR=/media/lext/FAST/OARSI_grading_project/workdir/oarsi_grades_snapshots_siamese/
DATA_DIR=/media/lext/FAST/OARSI_grading_project/Data/datasets/
META_DIR=/media/lext/FAST/OARSI_grading_project/
SNP_PREF=2019_03

cd scripts

python train.py --siamese True --unfreeze_epoch 2 --lr 1e-4 --lr_drop 30 --dropout_rate 0.2 \
                --val_bs 32 --siamese_bb resnet-18 --snapshots ${SNAPSHOTS_DIR} \
                --dataset_root ${DATA_DIR} --meta_root ${META_DIR}




for SNP_NAME in $(ls ${SNAPSHOTS_DIR} | grep ${SNP_PREF});
do
    python oof_inference.py --snapshots ${SNAPSHOTS_DIR} \
        --dataset_root ${DATA_DIR} \
        --meta_root ${META_DIR}\
        --snapshot ${SNP_NAME}
done

