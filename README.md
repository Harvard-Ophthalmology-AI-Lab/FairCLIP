
# FairCLIP

This repository provides the code and the dataset for the paper entitled *FairCLIP: Harnessing Fairness in Vision-Language Learning*. 

## Abstract

Fairness is a critical concern in deep learning, especially in healthcare, where these models influence diagnoses and treatment decisions. Although fairness has been investigated in the vision-only domain, the fairness of medical vision-language (VL) models remains unexplored due to the scarcity of medical VL datasets for studying fairness. To bridge this research gap, we introduce the first fair vision-language medical dataset FairVLMed that provides detailed demographic attributes, ground-truth labels, and clinical notes to facilitate an in-depth examination of fairness within VL foundation models. Using FairVLMed, we conduct a comprehensive fairness analysis of two widely-used VL models (CLIP and BLIP2), pre-trained on both natural and medical domains, across four different protected attributes. Our results highlight significant biases in all VL models, with Asian, Male, Non-Hispanic, and Spanish being the preferred subgroups across the protected attributes of race, gender, ethnicity, and language, respectively. In order to alleviate these biases, we propose FairCLIP, an optimal-transport-based approach that achieves a favorable trade-off between performance and fairness by reducing the Sinkhorn distance between the overall sample distribution and the distributions corresponding to each demographic group.


## Installation

To set up the required environment:

```bash
conda env create -f fairclip.yml
```

## Dataset

The FairVLMed dataset can be accessed via this [link](https://drive.google.com/drive/folders/1bkeifigwOAfnsLvup9mJOSNeA3WsvA2l?usp=sharing). This dataset can only be used for non-commercial research purposes. At no time, the dataset shall be used for clinical decisions or patient care. The data use license is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

The FairVLMed dataset comprises 10,000 samples from 10,000 subjects. It is divided into 7,000 training, 1,000 validation, and 2,000 test samples. Upon downloading and extracting these datasets, you will find the dataset structure as follows.

```
FairVLMed
├── data_summary.csv
├── gpt4_summarized_notes.csv
├── original_notes.csv
├── Testing
├── Training
└── Validation
```
The file split_files.csv details the division of data into training, validation, and testing sets. The data folder contains 10,000 NPZ files named in the format "data_xxxxx.npz", where "xxxxx" (e.g., 06691) is a unique numeric ID. The file meta_all.csv provides metadata (such as race, gender, ethnicity, marital status, age, and preferred language) for each NPZ file. Moreover, the files original_notes.csv and gpt4_summarized_notes.csv contain original notes and notes summarized by GPT-4, respectively.

Each NPZ file has the following fields.
```
slo_fundus: slo fundus image
md: visual field mean deviation
tds: 52 visual field total deviation values
age: patient age
gender: Female (0), Male (1)
race: Asian (0), Black (1), White (2)
ethnicity: non-Hispanic (0), Hispanic (1), Unknown (-1)
language: English (0), Spanish (1), Other (2), Unknown (-1)
maritalstatus: Marriage or Partnered (0), Single (1), Divorced (2), Widoled (3), Legally Separated (4), Unknown (-1)
glaucoma: Non-Glaucoma (0) or Glaucoma (1)
note: the original de-identified clinical note
```

## LLM Summarization
We use the following LLMs for summarizing the medical notes.
1. PMC-LLAMA
2. MED42
3. GPT-4

```bash
python src/dataset_deidentification_summarization.py --openai_key <YOUR_OPENAI_KEY> --models gpt-4
```

NOTE: OPENAI_KEY is only needed for GPT-4.

## Pre-training

<!-- ### MAE

```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --master_port=29501 --nproc_per_node=8 main_pretrain.py --batch_size 64 --model mae_vit_large_patch16 --norm_pix_loss --mask_ratio 0.75 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --data_path ${DATA_DIR} --output_dir $EXP_NAME --log_dir $EXP_NAME > ${EXP_NAME}.out
``` -->

### CLIP/FairCLIP
The code for pre-training **CLIP** and **FairCLIP** is in the folder [FairCLIP](./FairCLIP).

### BLIP-2
```bash
cd FairCLIP/LAVIS
python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 train.py --cfg-path FairCLIP/LAVIS/lavis/projects/blip2/train/pretrain_stage1.yaml
```

## Evaluation

### Zero-shot

### CLIP

```bash
python src/clip_eval.py
```

### BLIP-2

```bash
python src/blip_eval.py
```

### Linear Probing
For linear probing, use the following command. 

```bash
cd FairCLIP/mae
DATA_DIR=/Path/to/FairVLMed
FEATS_TYPE=image # [image, multimodal]

PRETRAIN_CHKPT=clip_vitl14_ep004.pth
EXP_NAME=tmp
MODEL_TYPE=clip # [clip, blip2]

OMP_NUM_THREADS=1 python -m torch.distributed.launch --master_port=29501 --nproc_per_node=1 main_linprobe.py --model_type ${MODEL_TYPE} --vl_feats_type ${FEATS_TYPE} --cfg-path FairCLIP/LAVIS/lavis/projects/blip2/train/pretrain_stage1.yaml --vision_encoder_weights clip --summary_type original --batch_size 512 --model vit_large_patch16 --cls_token --finetune ${PRETRAIN_CHKPT} --epochs 1000 --blr 0.1 --weight_decay 0.0 --data_path ${DATA_DIR} --output_dir $EXP_NAME --log_dir $EXP_NAME --nb_classes 2 > ${EXP_NAME}.out
```

## Citation

If you find our code or the FairVLMed dataset are helpful for your research, please consider citing our paper:

```
@InProceedings{Luo_2024_CVPR,
    author    = {Yan Luo, Min Shi, Muhammad Osama Khan, Muhammad Muneeb Afzal, Hao Huang, Shuaihang Yuan, Yu Tian, Luo Song, Ava Kouhana, Tobias Elze, Yi Fang, Mengyu Wang},
    title     = {airCLIP: Harnessing Fairness in Vision-and-Language Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {}
}

```

<!-- 
## Pre-trained Models

Download links for our pre-trained models can be found [here](LINK_TO_PRETRAINED_MODELS).

## Citation

If you find our work useful, please consider citing:

## License

This project is licensed under the terms of the [TBD License](LICENSE).

## Contact

For any queries, please feel free to open a GitHub issue.
-->