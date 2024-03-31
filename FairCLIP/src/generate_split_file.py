import os
import numpy as np
import random
import pandas as pd

input_folder = 'PATH-TO-DATASET/FairVLMed'

ratios = [0.7, 0.1, 0.2]

seed = random.randint(0, 100)
np.random.seed(seed)
random.seed(seed)

output_file = f'split_files_seed{seed}.csv'

# iterate over all files in the input_folder, not including subdirectories
all_files = []
for file in os.listdir(os.path.join(input_folder, 'data')):
    # check if the extension of the file is .npz
    if file.endswith('.npz'):
        all_files.append(file)

# shuffle the list of files
np.random.shuffle(all_files)

# split the list of files into train, val, and test
train_files = all_files[:int(ratios[0]*len(all_files))]
val_files = all_files[int(ratios[0]*len(all_files)):int((ratios[0]+ratios[1])*len(all_files))+1]
test_files = all_files[int((ratios[0]+ratios[1])*len(all_files))+1:]
print('Number of train files: {}'.format(len(train_files)))
print('Number of val files: {}'.format(len(val_files)))
print('Number of test files: {}'.format(len(test_files)))

data = {
    "filename": train_files+val_files+test_files,
    "file_type": ['train']*len(train_files)+['val']*len(val_files)+['test']*len(test_files)
}

# Create a DataFrame
df = pd.DataFrame(data)

df.to_csv(os.path.join(input_folder, output_file), index=False)


# read the csv file and construct lists of train, val, and test files
df = pd.read_csv(os.path.join(input_folder, output_file))
train_files_ = df[df['file_type'] == 'train']['filename'].tolist()
val_files_ = df[df['file_type'] == 'val']['filename'].tolist()
test_files_ = df[df['file_type'] == 'test']['filename'].tolist()

# check if train_files is the same as train_files_
print('Train: {}'.format(train_files == train_files_))
print('Val: {}'.format(val_files == val_files_))
print('Test: {}'.format(test_files == test_files_))