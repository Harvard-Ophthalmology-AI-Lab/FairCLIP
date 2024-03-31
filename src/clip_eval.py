import os
import clip
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

def zero_shot_clip(dataset, weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load('ViT-L/14', device)
    if weights:
        model.load_state_dict(torch.load(weights)['model'])

    predictions = []
    labels = []
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in dataset.classes]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        prediction = similarity[0].topk(1).indices.item()
        
        predictions.append(prediction)
        labels.append(label)

    predictions = np.array(predictions)
    labels = np.array(labels)
    accuracy = np.mean((labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")

def get_features_clip(dataset, model, device):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            image_features = model.encode_image(images.to(device))

            all_features.append(image_features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

def linear_probe_clip(train, test, weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load('ViT-L/14', device)
    if weights:
        model.load_state_dict(torch.load(weights)['model'])

    train.transform = preprocess
    test.transform = preprocess

    train_features, train_labels = get_features_clip(train, model, device)
    test_features, test_labels = get_features_clip(test, model, device)

    # Perform logistic regression
    # NOTE: the C value should be determined via a hyperparameter sweep using a validation split.
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")

if __name__ == "__main__":
    CACHE_DIR = ''
    # Zero-shot
    dataset = CIFAR100(root=os.path.expanduser(CACHE_DIR), download=True, train=False)
    zero_shot_clip(dataset)

    # Linear Probe
    root = os.path.expanduser(CACHE_DIR)
    train = CIFAR100(root, download=True, train=True)
    test = CIFAR100(root, download=True, train=False)
    linear_probe_clip(train, test)