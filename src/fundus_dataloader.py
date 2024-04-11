import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataloader import default_collate
import pandas as pd
# from lavis.processors.blip_processors import Blip2ImageTrainProcessor, BlipCaptionProcessor

class FUNDUS_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='', subset='train', vis_processor=None, text_processor=None, summary_type=None):
        if subset == 'train':
            self.dataset_dir = os.path.join(dataset_dir, 'Training')
        elif subset == 'val':
            self.dataset_dir = os.path.join(dataset_dir, 'Validation')
        elif subset == 'test':
            self.dataset_dir = os.path.join(dataset_dir, 'Test')
        else:
            raise Exception('invalid subset specified')
        self.subset = subset
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.files = [f for f in os.listdir(self.dataset_dir) if f.endswith('.npz')]

        self.summary_type = summary_type
        if self.summary_type == 'original':
            self.summary_file = pd.read_csv(os.path.join(dataset_dir, f'original_notes.csv'))
        else:
            self.summary_file = pd.read_csv(os.path.join(dataset_dir, f'{self.summary_type}_summarized_notes.csv'))

        # iterate through the files and remove the ones that have unknown attributes (-1)
        if self.subset != 'train':
            tmp_files = []
            for file in self.files:
                npz_path = os.path.join(self.dataset_dir, file)
                data = np.load(npz_path)
                if data['race'].item() != -1 and data['gender'].item() != -1 and data['ethnicity'].item() != -1 and data['language'].item() != -1:
                    tmp_files.append(file)
            self.files = tmp_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = os.path.join(self.dataset_dir, self.files[idx])
        data = np.load(npz_path)
        image = data['slo_fundus'].astype(np.float32)
        image = Image.fromarray(image).convert("RGB")

        image = self.vis_processor(image)
        note = str(self.summary_file.loc[self.summary_file['filename'] == self.files[idx]].iloc[:, 2].item()).strip()
        caption = self.text_processor(note)

        return {
            "image": image,
            "text_input": caption,
            "glaucoma": int(data['glaucoma'].item()),
            "attributes": torch.tensor([int(data['race'].item()), int(data['gender'].item()), int(data['ethnicity'].item()), int(data['language'].item())]),
        }

    def collater(self, samples):
        return default_collate(samples)

if __name__ == "__main__":
    dataset_dir = '../FUNDUS_Dataset/FairVLMed'
    subset = 'train'
    vis_processor = Blip2ImageTrainProcessor(image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0)
    text_processor = BlipCaptionProcessor(prompt='', max_words=50) # default settings
    # summary_type = 'original'
    summary_type = 'gpt-4'
    dset = FUNDUS_Dataset(dataset_dir, subset, vis_processor, text_processor, summary_type)
    print(len(dset))
    dset[0]
