#!/bin/bash
DATASET_DIR=/shared/ssd_30T/luoy/project/python/datasets/harvard/FairVLMed_publish
RESULT_DIR=.
MODEL_ARCH=vit-b16  # Options: vit-b16 | vit-l14
MODALITY_TYPE='slo_fundus'
LR=1e-5
BATCH_SIZE=32

RESULT_DIR='path/to/checkpoints/dir'  # Change this to your actual result directory
PERF_FILE=${RESULT_DIR}/CLIP_${MODEL_ARCH}_${MODALITY_TYPE}.csv

python ./evaluate_CLIP.py \
		--dataset_dir ${DATASET_DIR} \
		--result_dir ${RESULT_DIR} \
		--lr ${LR} \
		--perf_file ${PERF_FILE} \
		--model_arch ${MODEL_ARCH} \
		--pretrained_weights ${RESULT_DIR}/checkpoint_name.pth