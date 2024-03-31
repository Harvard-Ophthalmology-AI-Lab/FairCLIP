# Resource:
# https://github.com/openai/CLIP/issues/83
# https://github.com/openai/CLIP
# https://github.com/mlfoundations/wise-ft
# https://github.com/LightDXY/FT-CLIP/tree/main
# https://github.com/damian0815/finetune-clip-huggingface/tree/main
import os
import numpy as np
import random
from PIL import Image
import math
import copy
import pandas as pd
import re

import clip

import torch
import torch.nn as nn

from torchvision.models import *
import torch.nn.functional as F

from sklearn.metrics import *
from fairlearn.metrics import *

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def find_all_files(folder, suffix='npz'):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and os.path.join(folder, f).endswith(suffix)]
    return files

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        return self.model.encode_text(text)
    
class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model
        
    def forward(self,image):
        return self.model.encode_image(image)

def truncate_note(note, max_length=180):

    # truncate the note if it is longer than 77 words, but maintain the word integrity
    if len(note) > max_length:
        note = note[:max_length]
        note = note[:note.rfind(' ')]
    
    return note

def count_number_of_groups(input_dataset):
    instances_on_race = []
    instances_on_gender = []
    instances_on_ethnicity = []
    for file in input_dataset.files:
            npz_path = os.path.join(input_dataset.dataset_dir, file)
            data = np.load(npz_path)
            instances_on_race.append(data['race'].item())
            instances_on_gender.append(data['gender'].item())
            instances_on_ethnicity.append(data['ethnicity'].item())
    # count the unique number in instances_on_race
    _, numbers_of_race = np.unique(instances_on_race, return_counts=True)
    _, numbers_of_gender = np.unique(instances_on_gender, return_counts=True)
    _, numbers_of_ethnicity = np.unique(instances_on_ethnicity, return_counts=True)
    return numbers_of_race, numbers_of_gender, numbers_of_ethnicity


class fair_vl_med_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='', preprocess=None, files=None, subset='Training', text_source='note', summarized_note_file=None, ruleout_unknown=False):
        self.preprocess = preprocess
        self.dataset_dir = os.path.join(dataset_dir, subset)
        self.subset = subset
        self.text_source = text_source
        self.ruleout_unknown = ruleout_unknown

        self.summarized_notes = {}
        # summarized_note_file is a csv file that contains the summarized notes associated with npz files
        # read the summarized notes from the csv file and construct a dictionary
        if self.subset == 'Training' and self.text_source == 'note' and summarized_note_file != '':
            df = pd.read_csv(os.path.join(dataset_dir, summarized_note_file))
            
            for index, row in df.iterrows():
                self.summarized_notes[row.iloc[0].strip()] = row.iloc[2].strip()
        
        # check if the split file exists
        if files is not None:
            self.files = files
        else:
            # df = pd.read_csv(os.path.join(dataset_dir, 'split_files.csv'))
            # self.files = df[df['file_type'] == subset]['filename'].tolist()
            self.files = find_all_files(self.dataset_dir, suffix='npz')

        # iterate through the files and remove the ones that has unknown attributes (-1)
        if subset != 'Training' or self.ruleout_unknown:
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
        slo_fundus = data['slo_fundus'].astype(np.float32) # original size: (664, 512)
        slo_fundus = self.preprocess(Image.fromarray(slo_fundus))

        if self.subset == 'Training':
            if self.text_source == 'note':

                note = self.summarized_notes[self.files[idx]].strip()

                note = truncate_note(note)
                token = clip.tokenize(note)
                token = token.squeeze()
            elif self.text_source == 'label':
                glaucoma_label = int(data['glaucoma'].item())
                if glaucoma_label == 1:
                    note = 'A photo of glaucoma'
                else:
                    note = 'A photo of non-glaucoma'
                token = clip.tokenize(note)
                token = token.squeeze()
        else:
            note = 'A photo of non-glaucoma'
            neg_token = clip.tokenize(note)

            note = 'A photo of glaucoma'
            pos_token = clip.tokenize(note)

            # concatenate two tensors together, the final tensor will be at size of 2, 77
            token = torch.cat((neg_token, pos_token), dim=0)

        # extract glaucoma label from npz file
        glaucoma_label = int(data['glaucoma'].item())
        race = int(data['race'].item())
        gender = int(data['gender'].item())
        hispanic = int(data['ethnicity'].item())
        language = int(data['language'].item())
        # merge all labels together into a single tensor at size of 4
        label_and_attributes = torch.tensor([glaucoma_label, race, gender, hispanic, language])

        return slo_fundus, token, label_and_attributes

class fair_vl_group_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='', preprocess=None, files=None, subset='Training', text_source='note', summarized_note_file=None, attribute='race', thegroup=0):
        self.preprocess = preprocess
        self.dataset_dir = os.path.join(dataset_dir, subset)
        self.subset = subset
        self.text_source = text_source

        self.summarized_notes = {}
        # summarized_note_file is a csv file that contains the summarized notes associated with npz files
        # read the summarized notes from the csv file and construct a dictionary
        if self.subset == 'Training' and self.text_source == 'note' and summarized_note_file != '':
            df = pd.read_csv(os.path.join(dataset_dir, summarized_note_file))
            
            for index, row in df.iterrows():
                self.summarized_notes[row.iloc[0].strip()] = row.iloc[2].strip()
        
        # check if the split file exists
        if files is not None:
            self.files = files
        else:
            # df = pd.read_csv(os.path.join(dataset_dir, 'split_files.csv'))
            # self.files = df[df['file_type'] == subset]['filename'].tolist()
            self.files = find_all_files(self.dataset_dir, suffix='npz')

        # df = pd.read_csv(os.path.join(dataset_dir, 'split_files.csv'))
        # self.files = df[df['file_type'] == subset]['filename'].tolist()
        # self.files = files

        # iterate through the files and remove the ones that has unknown attributes (-1)
        if subset != 'Training':
            tmp_files = []
            for file in self.files:
                npz_path = os.path.join(self.dataset_dir, file)
                data = np.load(npz_path)
                if data['race'].item() != -1 and data['gender'].item() != -1 and data['ethnicity'].item() != -1 and data['language'].item() != -1:
                    tmp_files.append(file)
            self.files = tmp_files

        tmp_files = []
        for file in self.files:
            npz_path = os.path.join(self.dataset_dir, file)
            data = np.load(npz_path)

            group = int(data[attribute].item())
            if group == thegroup:
                tmp_files.append(file)
        self.files = tmp_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = os.path.join(self.dataset_dir, self.files[idx])
        data = np.load(npz_path)
        slo_fundus = data['slo_fundus'].astype(np.float32) # original size: (664, 512)
        slo_fundus = self.preprocess(Image.fromarray(slo_fundus))

        if self.subset == 'Training':
            if self.text_source == 'note':
                
                note = self.summarized_notes[self.files[idx]].strip()
                note = truncate_note(note)
                token = clip.tokenize(note)
                token = token.squeeze()
            elif self.text_source == 'label':
                glaucoma_label = int(data['glaucoma'].item())
                if glaucoma_label == 1:
                    note = 'A photo of glaucoma'
                else:
                    note = 'A photo of non-glaucoma'
                token = clip.tokenize(note)
                token = token.squeeze()
        else:
            note = 'A photo of non-glaucoma'
            neg_token = clip.tokenize(note)

            note = 'A photo of glaucoma'
            pos_token = clip.tokenize(note)

            # concatenate two tensors together, the final tensor will be at size of 2, 77
            token = torch.cat((neg_token, pos_token), dim=0)

        # extract glaucoma label from npz file
        glaucoma_label = int(data['glaucoma'].item())
        race = int(data['race'].item())
        gender = int(data['gender'].item())
        hispanic = int(data['ethnicity'].item())
        language = int(data['language'].item())
        # merge all labels together into a single tensor at size of 4
        label_and_attributes = torch.tensor([glaucoma_label, race, gender, hispanic, language])

        return slo_fundus, token, label_and_attributes

def endless_loader(dataloader):
    while True:
        for data in dataloader:
            yield data

class image_title_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='', preprocess=None, files=None, subset='train'):
        self.preprocess = preprocess
        self.files = files
        self.dataset_dir = dataset_dir
        self.subset = subset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = os.path.join(self.dataset_dir, self.files[idx])
        data = np.load(npz_path)
        slo_fundus = data['slo_fundus'].astype(np.float32)
        slo_fundus = self.preprocess(Image.fromarray(slo_fundus))

        if self.subset == 'train':
            note = truncate_note(data['note'].item().strip())
            token = clip.tokenize(note)
            token = token.squeeze()
        else:
            note = 'A photo of non-glaucoma'
            neg_token = clip.tokenize(note)

            note = 'A photo of glaucoma'
            pos_token = clip.tokenize(note)

            # concatenate two tensors together, the final tensor will be at size of 2, 77
            token = torch.cat((neg_token, pos_token), dim=0)

        # extract glaucoma label from npz file
        glaucoma_label = int(data['glaucoma'].item())
        race = int(data['race'].item())
        gender = int(data['gender'].item())
        hispanic = int(data['hispanic'].item())
        # merge all labels together into a single tensor at size of 4
        label_and_attributes = torch.tensor([glaucoma_label, race, gender, hispanic])


        return slo_fundus, token, label_and_attributes

class Adversary_Net(nn.Module):

    def __init__(self, n_sensitive, n_hidden=32):
        super(Adversary_Net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_sensitive),
        )

    def forward(self, x):
        return self.network(x)

def compute_vl_prob(img_feats, class_txt_feats):
    # img_feats: [batch_size, 512]
    # class_txt_feats: [batch_size, num_class, 512]

    all_logits = []
    for i in range(class_txt_feats.shape[1]):
        similarity = (img_feats @ class_txt_feats[:, i, :].T)
        # extract the diagonal of the matrix
        logits = similarity.diag()
        all_logits.append(logits)

    all_logits = torch.stack(all_logits, dim=1)

    # compute the probability by applying softmax along the second dimension
    vl_prob = torch.softmax(all_logits, dim=1)

    return vl_prob, all_logits

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if len(output.shape) == 1:
        acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
        return acc.item()
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, dim=1)
        target = target.view(batch_size, 1).repeat(1, maxk)
        
        correct = (pred == target)
  
        topk_accuracy = []
        for k in topk:
            accuracy = correct[:, :k].float().sum().item()
            accuracy /= batch_size # [0, 1.]
            topk_accuracy.append(accuracy)
        
        return topk_accuracy[0]

def compute_auc(pred_prob, y, num_classes=2):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        auc_val = auc(fpr, tpr)
    elif num_classes > 2:
        y_onehot = num_to_onehot(y, num_classes)
        auc_val = roc_auc_score(y_onehot, pred_prob, average='macro', multi_class='ovr')

    return auc_val

def auc_score(pred_prob, y):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if np.unique(y).shape[0]>2:
        AUC = roc_auc_score(y, pred_prob, multi_class='ovr')
    else:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        AUC = auc(fpr, tpr)
    
    return AUC

def num_to_onehot(nums, num_to_class):
    nums = nums.astype(int)
    n_values = num_to_class
    onehot_vec = np.eye(n_values)[nums]
    return onehot_vec

def prob_to_label(pred_prob):
    # Find the indices of the highest probabilities for each sample
    max_prob_indices = np.argmax(pred_prob, axis=1)

    # Create one-hot vectors for each sample
    one_hot_vectors = np.zeros_like(pred_prob)
    one_hot_vectors[np.arange(len(max_prob_indices)), max_prob_indices] = 1

    return one_hot_vectors

def numeric_to_one_hot(y, num_classes=None):
    y = np.asarray(y, dtype=np.int32)

    if num_classes is None:
        num_classes = np.max(y) + 1
    
    one_hot_array = np.zeros((len(y), num_classes))
    one_hot_array[np.arange(len(y)), y] = 1
    
    return one_hot_array

def multiclass_demographic_parity(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = demographic_parity_difference(pred_one_hot[:,i],
                                gt_one_hot[:,i],
                                sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)
        
    return avg_score

def multiclass_equalized_odds(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = equalized_odds_difference(pred_one_hot[:,i],
                            gt_one_hot[:,i],
                            sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)
        
    return avg_score

def multiclass_demographic_parity_(pred_prob, y, attrs):

    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    attrs_set = np.unique(attrs)
    y_pred = np.argmax(pred_prob, axis=1)

    mc_dpd = 0
    for i in range(pred_prob.shape[1]):
        tmp_preds = (y_pred==i).astype(int)
        tmp_not_preds = 1 - tmp_preds

        dp_by_attrs = []
        for j in attrs_set:
            idx = attrs==j
            tmp = np.abs(tmp_preds.mean().item() - tmp_preds[idx].mean().item()) + np.abs(tmp_not_preds.mean().item() - tmp_not_preds[idx].mean().item())
            dp_by_attrs.append(tmp)
            print(tmp)
        mc_dpd += np.mean(dp_by_attrs).item()

    mc_dpd = mc_dpd / pred_prob.shape[1]
        
    return mc_dpd

def auc_score_multiclass(pred_prob, y, num_of_class=3, eps=0.01):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    sensitivity_at_diff_specificity = [-1]*4
    y_onehot = num_to_onehot(y, num_of_class)
    fpr, tpr, thresholds = roc_curve(y_onehot.ravel(), pred_prob.ravel())
    for i in range(len(fpr)):
        cur_fpr = fpr[i]
        cur_tpr = tpr[i]
        if np.abs(cur_fpr-0.2) <= eps:
            sensitivity_at_diff_specificity[0] = cur_tpr
        if np.abs(cur_fpr-0.15) <= eps:
            sensitivity_at_diff_specificity[1] = cur_tpr
        if np.abs(cur_fpr-0.1) <= eps:
            sensitivity_at_diff_specificity[2] = cur_tpr
        if np.abs(cur_fpr-0.05) <= eps:
            sensitivity_at_diff_specificity[3] = cur_tpr
    AUC = auc(fpr, tpr)
    
    return AUC, sensitivity_at_diff_specificity

def equity_scaled_accuracy(output, target, attrs, alpha=1.):
    es_acc = 0
    if len(output.shape) >= 2:
        overall_acc = np.sum(np.argmax(output, axis=1) == target)/target.shape[0]
    else:
        overall_acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if len(pred_group.shape) >= 2:
            acc = np.sum(np.argmax(pred_group, axis=1) == gt_group)/gt_group.shape[0]
        else:
            acc = np.sum((pred_group >= 0.5).astype(float) == gt_group)/gt_group.shape[0]

        identity_wise_perf.append(acc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_acc)
    es_acc = (overall_acc / (alpha*tmp + 1))
    
    return es_acc

def equity_scaled_AUC(output, target, attrs, alpha=1., num_classes=2):
    es_auc = 0
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    
    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(target, output)
        overall_auc = auc(fpr, tpr)
    elif num_classes > 2:
        y_onehot = num_to_onehot(target, num_classes)
        overall_auc = roc_auc_score(y_onehot, output, average='macro', multi_class='ovr')

    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if num_classes == 2:
            fpr, tpr, thresholds = roc_curve(gt_group, pred_group)
            group_auc = auc(fpr, tpr)
        elif num_classes > 2:
            y_onehot = num_to_onehot(gt_group, num_classes)
            group_auc = roc_auc_score(y_onehot, pred_group, average='macro', multi_class='ovr')
        
        identity_wise_perf.append(group_auc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_auc)
    es_auc = (overall_auc / (alpha*tmp + 1))

    return es_auc

def evalute_perf_by_attr(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append( compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes) )
        aucs_by_attrs.append(aucs_by_group)
        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods


def evalute_comprehensive_perf(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    between_group_disparity = []

    overall_acc = accuracy(preds, gts, topk=(1,))
    overall_auc = compute_auc(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append( compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes) )
        aucs_by_attrs.append(aucs_by_group)
        std_disparity, max_disparity = compute_between_group_disparity(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return overall_acc, esaccs_by_attrs, overall_auc, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity

def evalute_comprehensive_perf_(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    between_group_disparity = []

    overall_auc = compute_auc(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append( compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes) )
        aucs_by_attrs.append(aucs_by_group)
        std_disparity, max_disparity = compute_between_group_disparity_half(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity

def compute_between_group_disparity(auc_list, overall_auc):
    return np.std(auc_list) / overall_auc, (np.max(auc_list)-np.min(auc_list)) / overall_auc

def compute_between_group_disparity_half(auc_list, overall_auc):
    return np.std(auc_list) / np.abs(overall_auc-0.5), (np.max(auc_list)-np.min(auc_list)) / np.abs(overall_auc-0.5)
