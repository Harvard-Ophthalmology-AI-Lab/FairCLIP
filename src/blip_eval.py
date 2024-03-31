import os
import numpy as np
import torch
from PIL import Image

from lavis.models import load_model_and_preprocess

from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from lavis.tasks.base_task import BaseTask
from lavis.common.config import Config
import argparse

from fundus_dataloader import FUNDUS_Dataset
from lavis.processors.blip_processors import BlipImageEvalProcessor, BlipCaptionProcessor

import sys
sys.path.append('../FairCLIP/src')
from modules import evalute_comprehensive_perf

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--vision_encoder_weights", type=str, required=True, choices=['clip', 'pmc-clip', 'fundus-clip', 'fundus-mae'])
    parser.add_argument("--vl_type", type=str, required=True, choices=['clip', 'blip', 'blip2'])
    parser.add_argument("--prompt", type=str, required=True, default='A picture of ') # a photo of a -- CLIP prompt
    parser.add_argument("--eval_type", type=str, required=True, choices=['zero_shot', 'linear_probe'])
    parser.add_argument("--summary_type", type=str, required=True, choices=['original', 'pmc-llama', 'med42', 'gpt-3.5-turbo', 'gpt-4'])

    args = parser.parse_args()
    return args

def zero_shot_blip(args):
    vl_type = args.vl_type
    prompt = args.prompt
    summary_type = args.summary_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = '../FUNDUS_Dataset/FairVLMed'
    subset = 'test'
    vis_processor = BlipImageEvalProcessor(image_size=224, mean=None, std=None)
    text_processor = BlipCaptionProcessor(prompt='', max_words=50) # default settings
    dataset = FUNDUS_Dataset(dataset_dir, subset, vis_processor, text_processor, summary_type)
    text_processor = BlipCaptionProcessor(prompt=prompt, max_words=50) # for creating prompts from class names

    if vl_type == "clip":
        model, vis_processors, text_processor = load_model_and_preprocess("clip_feature_extractor", model_type="ViT-L-14", is_eval=True, device=device)
    elif vl_type == "blip":
        model, vis_processors, _ = load_model_and_preprocess("blip_feature_extractor", model_type="base", is_eval=True, device=device)
    elif vl_type == "blip2":
        cfg = Config(args)
        cfg.model_cfg.pretrained = args.weights
        cfg.model_cfg.vision_encoder_weights = args.vision_encoder_weights
        cfg.model_cfg.load_pretrained = True
        model = BaseTask().build_model(cfg)
        model.to(device)
        model.eval()
    else:
        raise ValueError("Invalid vl_type. Choose either 'clip', 'blip' or 'blip2'")

    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu")
        state_dict = checkpoint["model"]
        msg = model.load_state_dict(state_dict, strict=False)
        print("Missing keys {}".format(msg.missing_keys))
        print("Number of Missing keys {}".format(len(msg.missing_keys)))
        print("load checkpoint from %s" % args.weights)
    
    predictions = []
    labels = []
    cls_prompt = [text_processor(cls_nm) for cls_nm in ['non-glaucoma', 'glaucoma']]
    
    all_probs = []
    all_labels = []
    all_attrs = []

    for idx in range(len(dataset)):
        batch = dataset[idx]
        image, label, attributes = batch['image'].unsqueeze(0).to(device), batch['glaucoma'], batch['attributes']
        sample = {"image": image, "text_input": cls_prompt}

        if vl_type == "clip":
            clip_features = model.extract_features(sample)
            image_features = clip_features.image_embeds_proj
            text_features = clip_features.text_embeds_proj
            sims = (image_features @ text_features.t())[0] / 0.01
        elif vl_type == "blip":
            image_features = model.extract_features(sample, mode="image").image_embeds_proj[:, 0]
            text_features = model.extract_features(sample, mode="text").text_embeds_proj[:, 0]
            sims = (image_features @ text_features.t())[0] / model.temp
        elif vl_type == "blip2":
            image_features = model.extract_features(sample, mode="image").image_embeds_proj
            text_features = model.extract_features(sample, mode="text").text_embeds_proj[:, 0]
            sims = (image_features @ text_features.t()).max(dim=1).values[0] / model.temp
        
        prediction = torch.nn.Softmax(dim=0)(sims).topk(1).indices.item()
        predictions.append(prediction)
        labels.append(label)
    
        all_probs.append(torch.nn.Softmax(dim=0)(sims)[1].detach().cpu())
        all_labels.append(torch.tensor(label).cpu())
        all_attrs.append(attributes.cpu())

    all_probs = torch.stack(all_probs).numpy()
    all_labels = torch.stack(all_labels).numpy()
    all_attrs = torch.stack(all_attrs).numpy()

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

    for k, v in test_stats.items():
        print(f"{k}: {v}")

    predictions = np.array(predictions)
    labels = np.array(labels)
    accuracy = np.mean((labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")

def get_features_blip(dataset, model, device, vl_type):
    all_features = []
    all_labels = []
    all_attrs = []
    
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size=100)):
            images, labels, attributes = batch['image'], batch['glaucoma'], batch['attributes']
            if vl_type == "clip":
                clip_features = model.extract_features({"image": images.to(device)})
                image_features = clip_features.image_embeds
            elif vl_type == "blip":
                image_features = model.extract_features({"image": images.to(device)}, mode="image").image_embeds[:, 0]
            elif vl_type == "blip2":
                image_features = model.extract_features({"image": images.to(device)}, mode="image").image_embeds.flatten(1)

            all_features.append(image_features)
            all_labels.append(labels)
            all_attrs.append(attributes)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy(), torch.cat(all_attrs).cpu().numpy()

def linear_probe_blip(args):
    vl_type = args.vl_type
    prompt = args.prompt
    summary_type = args.summary_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = '../FUNDUS_Dataset/FairVLMed'
    vis_processor = BlipImageEvalProcessor(image_size=224, mean=None, std=None)
    text_processor = BlipCaptionProcessor(prompt='', max_words=50) # default settings
    train = FUNDUS_Dataset(dataset_dir, 'train', vis_processor, text_processor, summary_type)
    test = FUNDUS_Dataset(dataset_dir, 'test', vis_processor, text_processor, summary_type)
    text_processor = BlipCaptionProcessor(prompt=prompt, max_words=50) # for creating prompts from class names

    if vl_type == "clip":
        model, vis_processors, text_processor = load_model_and_preprocess("clip_feature_extractor", model_type="ViT-L-14", is_eval=True, device=device)
    elif vl_type == "blip":
        model, vis_processors, _ = load_model_and_preprocess("blip_feature_extractor", model_type="base", is_eval=True, device=device)
    elif vl_type == "blip2":
        cfg = Config(args)
        cfg.model_cfg.pretrained = args.weights
        cfg.model_cfg.vision_encoder_weights = args.vision_encoder_weights
        cfg.model_cfg.load_pretrained = True
        model = BaseTask().build_model(cfg)
        model.to(device)
        model.eval()
    else:
        raise ValueError("Invalid vl_type. Choose either 'clip', 'blip' or 'blip2'")
        
    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu")
        state_dict = checkpoint["model"]
        msg = model.load_state_dict(state_dict, strict=False)
        print("Missing keys {}".format(msg.missing_keys))
        print("Number of Missing keys {}".format(len(msg.missing_keys)))
        print("load checkpoint from %s" % args.weights)

    train_features, train_labels, train_attrs = get_features_blip(train, model, device, vl_type)
    test_features, test_labels, test_attrs = get_features_blip(test, model, device, vl_type)

    # Perform logistic regression
    # NOTE: the C value should be determined via a hyperparameter sweep using a validation split.
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=10000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")

    overall_acc, eval_es_acc, overall_auc, eval_es_auc, eval_aucs_by_attrs, eval_dpds, eval_eods, between_group_disparity = evalute_comprehensive_perf(classifier.predict_proba(test_features)[:,1], test_labels, test_attrs.T)
    
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

    for k, v in test_stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    args = parse_args()
    if args.eval_type == "zero_shot":
        zero_shot_blip(args)
    elif args.eval_type == "linear_probe":
        linear_probe_blip(args)


