#!/usr/bin/env bash

SNAPSHOTS_DIR=/media/lext/FAST/OARSI_grading_project/workdir/oarsi_grades_snapshots_weighing_exp/
DATA_DIR=/media/lext/FAST/OARSI_grading_project/Data/datasets/
META_DIR=/media/lext/FAST/OARSI_grading_project/
SNP_PREF=2019_02

cd scripts


# -------------------------------------------------------------- #
# --------------------------No Weighing------------------------- #
# -------------------------------------------------------------- #

#python train.py --backbone_depth 18 --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR}

#python train.py --backbone_depth 34 --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR}

#python train.py --backbone_depth 50 --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR}

#python train.py --backbone_depth 50 --se True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR}

#python train.py --backbone_depth 50 --se True --dw True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR}

# -------------------------------------------------------------- #
# --------------------KL-based Weighing------------------------- #
# -------------------------------------------------------------- #

#python train.py --backbone_depth 18 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR}

#python train.py --backbone_depth 34 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR}

#python train.py --backbone_depth 50 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR}

#python train.py --backbone_depth 50 --se True  --weighted_sampling True --snapshots ${SNAPSHOTS_DIR}\
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR}

#python train.py --backbone_depth 50 --se True --dw True  --weighted_sampling True --snapshots ${SNAPSHOTS_DIR}\
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR}


# -------------------------------------------------------------- #
# --------------------Maximum task-weighing---------------------- #
# -------------------------------------------------------------- #

#python train.py --backbone_depth 18 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR} --mtw True

#python train.py --backbone_depth 34 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR}  --mtw True

#python train.py --backbone_depth 50 --weighted_sampling True --snapshots ${SNAPSHOTS_DIR} \
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR} --mtw True

#python train.py --backbone_depth 50 --se True  --weighted_sampling True --snapshots ${SNAPSHOTS_DIR}\
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR} --mtw True

#python train.py --backbone_depth 50 --se True --dw True  --weighted_sampling True --snapshots ${SNAPSHOTS_DIR}\
# --dataset_root ${DATA_DIR} --meta_root ${META_DIR} --mtw True



for SNP_NAME in $(ls ${SNAPSHOTS_DIR} | grep ${SNP_PREF});
do
    python oof_inference.py --snapshots ${SNAPSHOTS_DIR} \
        --dataset_root ${DATA_DIR} \
        --meta_root ${META_DIR}\
        --snapshot ${SNP_NAME}
done

