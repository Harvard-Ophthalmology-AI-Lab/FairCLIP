# FairCLIP

This repository provides the code and the dataset for the paper entitled *FairCLIP: Harnessing Fairness in Vision-Language Learning*. 

## Dataset

The FairVLMed dataset can be accessed via this [link](https://drive.google.com/drive/folders/1bkeifigwOAfnsLvup9mJOSNeA3WsvA2l?usp=sharing). This dataset can only be used for non-commercial research purposes. At no time, the dataset shall be used for clinical decisions or patient care. The data use license is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

The FairVLMed dataset comprises 10,000 samples from 10,000 subjects. It is divided into 7,000 training, 1,000 validation, and 2,000 test samples. Upon downloading and extracting these datasets, you will find the dataset structure as follows.

```
FairVLMed
├── data_summary.csv
├── gpt-4_summarized_notes.csv
├── original_notes.csv
├── Test
├── Training
└── Validation
```
The file split_files.csv details the division of data into training, validation, and test sets. The data folder contains 10,000 NPZ files named in the format "data_xxxxx.npz", where "xxxxx" (e.g., 06691) is a unique numeric ID. The file meta_all.csv provides metadata (such as race, gender, ethnicity, marital status, age, and preferred language) for each NPZ file. Moreover, the files original_notes.csv and gpt-4_summarized_notes.csv contain original notes and notes summarized by GPT-4, respectively.

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


## Requirements

To install the prerequisites, run:

```
pip install - r requirements.txt
```

## Experiments

To run the experiments for zero-shot transfer with CLIP, execute:

```
./scripts/finetune_CLIP.sh
```

To run the experiments for zero-shot transfer with FairCLIP, execute:

```
./scripts/finetune_FairCLIP.sh
```

To evaluate the models pre-trained in the above processes, execute:

```
./scripts/evaluate_CLIP.sh
```
