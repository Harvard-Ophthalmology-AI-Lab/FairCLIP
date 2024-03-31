# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np

import sys
sys.path.append('../FairCLIP/src')
from modules import evalute_comprehensive_perf

import clip

def truncate_note(note, max_length=170):
    # truncate the note if it is longer than 77 words, but maintain the word integrity
    if len(note) > max_length:
        note = note[:max_length]
        note = note[:note.rfind(' ')]
    return note

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = batch['image']
        targets = batch['glaucoma']
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if args.model_type in ['vit', 'mae']:
                outputs = model(samples)
            elif args.model_type == 'blip2':
                if args.vl_feats_type == 'image':
                    blip_feats = model.module.extract_features({"image": samples}, mode="image").image_embeds
                    if args.blip_feats_select == 'first':
                        blip_feats = blip_feats[:,0,:]
                    elif args.blip_feats_select == 'avgpool':
                        blip_feats = blip_feats.mean(dim=1)
                    elif args.blip_feats_select == 'maxpool':
                        blip_feats = blip_feats.max(dim=1)[0]
                    elif args.blip_feats_select == 'flatten':
                        blip_feats = blip_feats.flatten(1)
                elif args.vl_feats_type == 'multimodal':
                    assert args.blip_feats_select == 'avgpool'
                    blip_feats = model.module.extract_features({"image": samples, "text_input": batch['text_input']}, mode="multimodal").multimodal_embeds.mean(dim=1)
                outputs = model.module.head(blip_feats)
            elif args.model_type == 'clip':
                if args.vl_feats_type == 'image':
                    outputs = model.module.head(model.module.encode_image(samples))
                elif args.vl_feats_type == 'multimodal':
                    clip_text_input = torch.cat([clip.tokenize(truncate_note(tmp_note)) for tmp_note in batch['text_input']]).to(device)
                    concat_feats = torch.cat([model.module.encode_image(samples), model.module.encode_text(clip_text_input)], dim=1)
                    outputs = model.module.head(concat_feats)
            loss = torch.nn.BCEWithLogitsLoss()(outputs[:,1], targets.type(torch.float32))

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = None

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    all_probs = []
    all_labels = []
    all_attrs = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch['image']
        target = batch['glaucoma']
        attributes = batch['attributes']
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            if args.model_type in ['vit', 'mae']:
                output = model(images)
            elif args.model_type == 'blip2':
                if args.vl_feats_type == 'image':
                    blip_feats = model.module.extract_features({"image": images}, mode="image").image_embeds
                    if args.blip_feats_select == 'first':
                        blip_feats = blip_feats[:,0,:]
                    elif args.blip_feats_select == 'avgpool':
                        blip_feats = blip_feats.mean(dim=1)
                    elif args.blip_feats_select == 'maxpool':
                        blip_feats = blip_feats.max(dim=1)[0]
                    elif args.blip_feats_select == 'flatten':
                        blip_feats = blip_feats.flatten(1)
                elif args.vl_feats_type == 'multimodal':
                    assert args.blip_feats_select == 'avgpool'
                    blip_feats = model.module.extract_features({"image": images, "text_input": batch['text_input']}, mode="multimodal").multimodal_embeds.mean(dim=1)
                output = model.module.head(blip_feats)
            elif args.model_type == 'clip':
                if args.vl_feats_type == 'image':
                    output = model.module.head(model.module.encode_image(images))
                elif args.vl_feats_type == 'multimodal':
                    clip_text_input = torch.cat([clip.tokenize(truncate_note(tmp_note)) for tmp_note in batch['text_input']]).to(device)
                    concat_feats = torch.cat([model.module.encode_image(images), model.module.encode_text(clip_text_input)], dim=1)
                    output = model.module.head(concat_feats)
            loss = torch.nn.BCEWithLogitsLoss()(output[:,1], target.type(torch.float32))

        all_probs.append(torch.sigmoid(output)[:,1].cpu().numpy())
        all_labels.append(target.cpu().numpy())
        all_attrs.append(attributes.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_attrs = np.concatenate(all_attrs, axis=0)

    overall_acc, eval_es_acc, overall_auc, eval_es_auc, eval_aucs_by_attrs, eval_dpds, eval_eods, between_group_disparity = evalute_comprehensive_perf(all_probs, all_labels, all_attrs.T)
    
    test_stats = {
        'overall_acc': overall_acc,
        'eval_es_acc': eval_es_acc,
        'overall_auc': overall_auc,
        'eval_es_auc': eval_es_auc,
        'eval_aucs_by_attrs': eval_aucs_by_attrs,
        'eval_dpds': eval_dpds,
        'eval_eods': eval_eods,
        'between_group_disparity': between_group_disparity
    }

    return test_stats