
# FairCLIP
> [**CVPR 2024**] [**FairCLIP: Harnessing Fairness in Vision-Language Learning**](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_FairCLIP_Harnessing_Fairness_in_Vision-Language_Learning_CVPR_2024_paper.pdf)
>
> by [Yan Luo*](https://luoyan407.github.io/), [Min Shi*](https://shiminxst.github.io/index.html), Muhammad Osama Khan*, Muhammad Muneeb Afzal, Hao Huang, Shuaihang Yuan,  [Yu Tian](https://yutianyt.com/), Luo Song, Ava Kouhana, [Tobias Elze](http://www.tobias-elze.de/), [Yi Fang](https://engineering.nyu.edu/faculty/yi-fang), and [Mengyu Wang](https://ophai.hms.harvard.edu/team/dr-wang/).
>


## Abstract

Fairness is a critical concern in deep learning, especially in healthcare, where these models influence diagnoses and treatment decisions. Although fairness has been investigated in the vision-only domain, the fairness of medical vision-language (VL) models remains unexplored due to the scarcity of medical VL datasets for studying fairness. To bridge this research gap, we introduce the first fair vision-language medical dataset (*Harvard-FairVLMed*) that provides detailed demographic attributes, ground-truth labels, and clinical notes to facilitate an in-depth examination of fairness within VL foundation models. Using *Harvard-FairVLMed*, we conduct a comprehensive fairness analysis of two widely-used VL models (CLIP and BLIP2), pre-trained on both natural and medical domains, across four different protected attributes. Our results highlight significant biases in all VL models, with Asian, Male, Non-Hispanic, and Spanish being the preferred subgroups across the protected attributes of race, gender, ethnicity, and language, respectively. In order to alleviate these biases, we propose FairCLIP, an optimal-transport-based approach that achieves a favorable trade-off between performance and fairness by reducing the Sinkhorn distance between the overall sample distribution and the distributions corresponding to each demographic group.


## Installation

To set up the required environment:

```bash
conda create --name fairclip python=3.9.12
pip install -r requirements.txt
```

## Dataset

The Harvard-FairVLMed dataset can be accessed via this [link](https://drive.google.com/drive/folders/1bkeifigwOAfnsLvup9mJOSNeA3WsvA2l?usp=drive_link). This dataset can only be used for non-commercial research purposes. At no time, the dataset shall be used for clinical decisions or patient care. The data use license is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). If you have any questions, please email <harvardophai@gmail.com> and <harvardairobotics@gmail.com>.

Note that, the modifier word “Harvard” only indicates that our dataset is from the Department of Ophthalmology of Harvard Medical School and does not imply an endorsement, sponsorship, or assumption of responsibility by either Harvard University or Harvard Medical School as a legal identity.

The Harvard-FairVLMed dataset comprises 10,000 samples from 10,000 subjects. It is divided into 7,000 training, 1,000 validation, and 2,000 test samples. Upon downloading and extracting these datasets, you will find the dataset structure as follows.

```
Harvard-FairVLMed
├── data_summary.csv
├── gpt-4_summarized_notes.csv
├── Training
├── Validation
└── Test
```
The file split_files.csv details the division of data into training, validation, and testing sets. The data folder contains 10,000 NPZ files named in the format "data_xxxxx.npz", where "xxxxx" (e.g., 06691) is a unique numeric ID. The file meta_all.csv provides metadata (such as race, gender, ethnicity, marital status, age, and preferred language) for each NPZ file. Moreover, the files original_notes.csv and gpt-4_summarized_notes.csv contain original notes and notes summarized by GPT-4, respectively.

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
note_extra: the original de-identified clinical note with demographic attributes placed at the beginning
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

### CLIP/FairCLIP
The code for pre-training **CLIP** and **FairCLIP** is in the folder [FairCLIP](./FairCLIP).

### BLIP-2
```bash
cd FairCLIP/LAVIS
python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1.yaml
```

## Evaluation

### Linear Probing
```bash
cd FairCLIP/mae
DATA_DIR=/Path/to/FairVLMed
FEATS_TYPE=image # [image, multimodal]

PRETRAIN_CHKPT=/Path/to/CKPT
EXP_NAME=tmp
MODEL_TYPE=blip2 # [clip, blip2]

OMP_NUM_THREADS=1 python -m torch.distributed.launch --master_port=29501 --nproc_per_node=1 main_linprobe.py --model_type ${MODEL_TYPE} --vl_feats_type ${FEATS_TYPE} --blip_feats_select avgpool --cfg-path ../LAVIS/lavis/projects/blip2/train/pretrain_stage1.yaml --vision_encoder_weights clip --summary_type original --batch_size 512 --model vit_large_patch16 --cls_token --finetune ${PRETRAIN_CHKPT} --epochs 1000 --blr 0.1 --weight_decay 0.0 --data_path ${DATA_DIR} --output_dir $EXP_NAME --log_dir $EXP_NAME --nb_classes 2 > ${EXP_NAME}.out
```

## Checkpoints

The checkpoints trained in the CUDA 12.1 environment (see requirements.txt) can be found at [this shared folder](https://drive.google.com/drive/folders/1byRoH6--ShErPrCliRqt8Fw6qofPihg5?usp=sharing). Their performance is comparable to prior versions, as shown below.

| Method (ViT-16)       | AUC    | ES-AUC (Race) | ES-AUC (Gender) | ES-AUC (Ethnicity) | Black AUC | Asian AUC | White AUC | Female AUC | Male AUC | Non-Hispanic AUC | Hispanic AUC |
|-----------------------|--------|---------------|------------------|---------------------|-----------|-----------|-----------|-------------|----------|-------------------|---------------|
| CLIP                  | 0.6902 | 0.6464        | 0.6495           | 0.6212              | 0.7242    | 0.7078    | 0.6741    | 0.6628      | 0.7253   | 0.6940            | 0.5830        |
| FairCLIP (Race)       | 0.7106 | 0.6714        | 0.6621           | 0.6141              | 0.7357    | 0.7306    | 0.6973    | 0.6786      | 0.7518   | 0.7154            | 0.5583        |
| FairCLIP (Gender)     | 0.7222 | 0.6698        | 0.6770           | 0.6443              | 0.7728    | 0.7378    | 0.7102    | 0.6923      | 0.7591   | 0.7260            | 0.6052        |
| FairCLIP (Ethnicity)  | 0.7275 | 0.6662        | 0.6786           | 0.6519              | 0.7697    | 0.7624    | 0.7126    | 0.6955      | 0.7676   | 0.7316            | 0.6156        |


## Acknowledgment and Citation

If you find this repository useful for your research, please consider citing our [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_FairCLIP_Harnessing_Fairness_in_Vision-Language_Learning_CVPR_2024_paper.pdf):

```bibtex
@inproceedings{luo2024fairclip,
  title={Fairclip: Harnessing fairness in vision-language learning},
  author={Luo, Yan and Shi, Min and Khan, Muhammad Osama and Afzal, Muhammad Muneeb and Huang, Hao and Yuan, Shuaihang and Tian, Yu and Song, Luo and Kouhana, Ava and Elze, Tobias and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12289--12301},
  year={2024}
}

```
