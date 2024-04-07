import os
import numpy as np
import random
import argparse
import time
import json
import pandas as pd
from collections import Counter
from geomloss import SamplesLoss

import clip

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

import sys
sys.path.append('.')

from src.modules import *
from src import logger


parser = argparse.ArgumentParser(description='FairCLIP Training/Fine-Tuning')

parser.add_argument('--seed', default=-1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')

parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--dataset_dir', default='./data', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--eval_set', default='test', type=str, help='options: val | test')
parser.add_argument('--summarized_note_file', default='', type=str)
parser.add_argument('--text_source', default='note', type=str, help='options: note | label')
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--model_arch', default='vit-b16', type=str, help='options: vit-b16 | vit-l14')
parser.add_argument('--pretrained_weights', default='', type=str)
parser.add_argument('--attribute', default='race', type=str, help='race|gender|ethnicity|language')
parser.add_argument('--batchsize_fairloss', default=64, type=int)
parser.add_argument('--lambda_fairloss', default=1e-4, type=float)
parser.add_argument('--sinkhorn_blur', default=1e-4, type=float)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)

    logger.log(f'===> random seed: {args.seed}')

    logger.configure(dir=args.result_dir, log_suffix='train')

    with open(os.path.join(args.result_dir, f'args_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # the number of groups in each attribute
    groups_in_attrs = [3, 2, 2, 3]
    attr_to_idx = {'race': 0, 'gender': 1, 'ethnicity': 2, 'language': 3}

    model_arch_mapping = {'vit-b16': 'ViT-B/16', 'vit-l14': 'ViT-L/14'}

    best_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'best_{args.perf_file}')
    acc_head_str = ''
    auc_head_str = ''
    dpd_head_str = ''
    eod_head_str = ''
    esacc_head_str = ''
    esauc_head_str = ''
    group_disparity_head_str = ''
    if args.perf_file != '':
        if not os.path.exists(best_global_perf_file):
            for i in range(len(groups_in_attrs)):
                auc_head_str += ', '.join([f'auc_attr{i}_group{x}' for x in range(groups_in_attrs[i])]) + ', '
            dpd_head_str += ', '.join([f'dpd_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            eod_head_str += ', '.join([f'eod_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            esacc_head_str += ', '.join([f'esacc_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            esauc_head_str += ', '.join([f'esauc_attr{x}' for x in range(len(groups_in_attrs))]) + ', '

            group_disparity_head_str += ', '.join([f'std_group_disparity_attr{x}, max_group_disparity_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            
            with open(best_global_perf_file, 'w') as f:
                f.write(f'epoch, acc, {esacc_head_str} auc, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_head_str} path\n')

    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model, preprocess = clip.load(model_arch_mapping[args.model_arch], device=device, jit=False) #Must set jit=False for training

    train_files = None
    test_files = None

    train_dataset = fair_vl_med_dataset(args.dataset_dir, preprocess, subset='Training', text_source=args.text_source, summarized_note_file=args.summarized_note_file)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    val_dataset = fair_vl_med_dataset(args.dataset_dir, preprocess, subset='Validation')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    test_dataset = fair_vl_med_dataset(args.dataset_dir, preprocess, subset='Test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    logger.log(f'# of training samples: {train_dataset.__len__()}, # of testing samples: {test_dataset.__len__()}')
    
    group_dataloaders = []
    for i in range(groups_in_attrs[attr_to_idx[args.attribute]]):
        tmp_dataset = fair_vl_group_dataset(args.dataset_dir, preprocess, 
                text_source='note', summarized_note_file=args.summarized_note_file, 
                attribute=args.attribute, thegroup=i)
        tmp_dataloader = DataLoader(tmp_dataset, batch_size=args.batchsize_fairloss, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=False)
        group_dataloaders.append(endless_loader(tmp_dataloader))

    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(train_dataset)
    logger.log(f'group size on race in training set: {group_size_on_race}')
    logger.log(f'group size on gender in training set: {group_size_on_gender}')
    logger.log(f'group size on ethnicity in training set: {group_size_on_ethnicity}')
    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(test_dataset)
    logger.log(f'group size on race in test set: {group_size_on_race}')
    logger.log(f'group size on gender in test set: {group_size_on_gender}')
    logger.log(f'group size on ethnicity in test set: {group_size_on_ethnicity}')

    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 


    if device == "cpu":
      model.float()
    else :
      clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": model.transformer.parameters(), "lr": args.lr},
        {"params": model.visual.parameters(), "lr": args.lr},
    ], lr=args.lr, betas=(0.1, 0.1), eps=1e-6,weight_decay=args.weight_decay)
    
    loss_for_FairCLIP = SamplesLoss(loss="sinkhorn", p=2, blur=args.sinkhorn_blur)

    # if args.pretrained_weights != "":
    #     checkpoint = torch.load(args.pretrained_weights)

    #     start_epoch = checkpoint['epoch'] + 1
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    best_epoch = 0
    best_loss = 1000000
    best_auc_groups = None
    best_acc_groups = None
    best_pred_gt_by_attr = None
    best_auc = sys.float_info.min
    best_acc = sys.float_info.min
    best_es_acc = sys.float_info.min
    best_es_auc = sys.float_info.min
    best_between_group_disparity = None

    for epoch in range(args.num_epochs):
        avg_loss = 0
        for batch in train_dataloader :
            optimizer.zero_grad()

            images, texts, label_and_attributes = batch 

            images= images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            
            similarity = (logits_per_image @ logits_per_text.T)
            correlations_with_batch = similarity.diag().float()
            correlations_groups = []
            
            for x in group_dataloaders:
                images_dist, texts_dist, label_and_attributes_dist = next(x)
                images_dist= images_dist.to(device)
                texts_dist = texts_dist.to(device)
                with torch.no_grad():
                    img_feats, txt_feats = model(images_dist, texts_dist)

                similarity = (img_feats @ txt_feats.T)
                correlations_with_group = similarity.diag().float()
                correlations_with_group /= correlations_with_group.sum()
                
                total_loss = total_loss + args.lambda_fairloss * loss_for_FairCLIP(correlations_with_batch[:,None], correlations_with_group[:,None])

            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            avg_loss += total_loss.item()
            


        avg_loss /= len(train_dataloader)
        
        # iterate over test dataset
        eval_avg_loss = 0
        all_probs = []
        all_labels = []
        all_attrs = []
        for batch in test_dataloader :
            images,texts, label_and_attributes = batch 

            images= images.to(device)
            texts = texts.to(device)
            glaucoma_labels = label_and_attributes[:, 0].to(device)
            attributes = label_and_attributes[:, 1:].to(device)

            class_text_feats = []
            with torch.no_grad():
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=1, keepdim=True)

                for i in range(texts.shape[1]):
                    text_features = model.encode_text(texts[:, i, :])
                    text_features /= text_features.norm(dim=1, keepdim=True)
                    class_text_feats.append(text_features[:, None, :])
                # concatentate class_text_feats along the second dimension
                class_text_feats = torch.cat(class_text_feats, dim=1)
                
            vl_prob, vl_logits = compute_vl_prob(image_features, class_text_feats)

            all_probs.append(vl_prob[:,1].cpu().numpy())
            all_labels.append(glaucoma_labels.cpu().numpy())
            all_attrs.append(attributes.cpu().numpy())

            # apply binary cross entropy loss
            loss = F.binary_cross_entropy(vl_prob[:,1].float(), glaucoma_labels.float())
            eval_avg_loss += loss.item()

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_attrs = np.concatenate(all_attrs, axis=0)
        eval_avg_loss /= len(test_dataloader)

        logger.log(f'===> epoch[{epoch:03d}/{args.num_epochs:03d}], training loss: {avg_loss:.4f}, eval loss: {eval_avg_loss:.4f}')

        overall_acc, eval_es_acc, overall_auc, eval_es_auc, eval_aucs_by_attrs, eval_dpds, eval_eods, between_group_disparity = evalute_comprehensive_perf(all_probs, all_labels, all_attrs.T)

        if best_auc <= overall_auc:
            best_auc = overall_auc
            best_acc = overall_acc
            best_ep = epoch
            best_auc_groups = eval_aucs_by_attrs
            best_dpd_groups = eval_dpds
            best_eod_groups = eval_eods
            best_es_acc = eval_es_acc
            best_es_auc = eval_es_auc
            best_between_group_disparity = between_group_disparity
        
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': eval_avg_loss,
                }, os.path.join(args.result_dir, f"clip_ep{epoch:03d}.pth"))


        if args.result_dir is not None:
            np.savez(os.path.join(args.result_dir, f'pred_gt_ep{epoch:03d}.npz'), 
                        val_pred=all_probs, val_gt=all_labels, val_attr=all_attrs)

        logger.log(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
        logger.log(f'---- best AUC by groups and attributes at epoch {best_ep}')
        logger.log(best_auc_groups)

        logger.logkv('epoch', epoch)
        logger.logkv('trn_loss', round(avg_loss,4))

        logger.logkv('eval_loss', round(eval_avg_loss,4))
        logger.logkv('eval_acc', round(overall_acc,4))
        logger.logkv('eval_auc', round(overall_auc,4))

        for ii in range(len(eval_es_acc)):
            logger.logkv(f'eval_es_acc_attr{ii}', round(eval_es_acc[ii],4))
        for ii in range(len(eval_es_auc)):
            logger.logkv(f'eval_es_auc_attr{ii}', round(eval_es_auc[ii],4))
        for ii in range(len(eval_aucs_by_attrs)):
            for iii in range(len(eval_aucs_by_attrs[ii])):
                logger.logkv(f'eval_auc_attr{ii}_group{iii}', round(eval_aucs_by_attrs[ii][iii],4))

        for ii in range(len(between_group_disparity)):
            logger.logkv(f'eval_auc_attr{ii}_std_group_disparity', round(between_group_disparity[ii][0],4))
            logger.logkv(f'eval_auc_attr{ii}_max_group_disparity', round(between_group_disparity[ii][1],4))

        for ii in range(len(eval_dpds)):
            logger.logkv(f'eval_dpd_attr{ii}', round(eval_dpds[ii],4))
        for ii in range(len(eval_eods)):
            logger.logkv(f'eval_eod_attr{ii}', round(eval_eods[ii],4))

        logger.dumpkvs()
    
    if args.perf_file != '':
        if os.path.exists(best_global_perf_file):
            with open(best_global_perf_file, 'a') as f:

                esacc_head_str = ', '.join([f'{x:.4f}' for x in best_es_acc]) + ', '
                esauc_head_str = ', '.join([f'{x:.4f}' for x in best_es_auc]) + ', '

                auc_head_str = ''
                for i in range(len(best_auc_groups)):
                    auc_head_str += ', '.join([f'{x:.4f}' for x in best_auc_groups[i]]) + ', '

                group_disparity_str = ''
                for i in range(len(best_between_group_disparity)):
                    group_disparity_str += ', '.join([f'{x:.4f}' for x in best_between_group_disparity[i]]) + ', '
                
                dpd_head_str = ', '.join([f'{x:.4f}' for x in best_dpd_groups]) + ', '
                eod_head_str = ', '.join([f'{x:.4f}' for x in best_eod_groups]) + ', '

                path_str = f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}'
                f.write(f'{best_ep}, {best_acc:.4f}, {esacc_head_str} {best_auc:.4f}, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_str} {path_str}\n')

    os.rename(args.result_dir, f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}')