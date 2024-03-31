#!/bin/bash
DATASET_DIR=/PATH-TO_DATASET/FairVLMed
RESULT_DIR=.
MODEL_ARCH=vit-b16 # Options: vit-b16 | vit-l14
NUM_EPOCH=10
MODALITY_TYPE='slo_fundus'
ATTRIBUTE_TYPE=race # Options: race | gender | ethnicity | language
SUMMARIZED_NOTE_FILE=gpt4_summarized_notes.csv
LR=1e-5
BATCH_SIZE=32

PERF_FILE=${MODEL_ARCH}_${MODALITY_TYPE}.csv

python ./finetune_CLIP.py \
		--dataset_dir ${DATASET_DIR} \
		--result_dir ${RESULT_DIR}/results/glaucoma_CLIP_${MODEL_ARCH} \
		--lr ${LR} \
		--batch_size ${BATCH_SIZE} \
		--perf_file ${PERF_FILE} \
		--model_arch ${MODEL_ARCH} \
		--summarized_note_file ${SUMMARIZED_NOTE_FILE} 