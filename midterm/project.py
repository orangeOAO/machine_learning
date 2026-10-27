"""TrashNet experiment. Run prepare, train, evaluate, then demo.py.

All paths are relative to this file, so commands work from any directory.
"""
import argparse
import csv
import hashlib
import json
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
LABELS = ['紙板', '玻璃', '金屬', '紙張', '塑膠', '其他垃圾']
MODEL_NAMES = ['cnn', 'resnet18', 'dnn_tc']
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def prepare():
    DATA.mkdir(exist_ok=True)
    archive = DATA / 'dataset-resized.zip'
    url = 'https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (DATA / member.filename).resolve()
            if not target.is_relative_to(DATA.resolve()):
                raise ValueError('Unsafe archive path')
        z.extractall(DATA)
    rows, seen, duplicates, conflicts = [], {}, [], set()
    for label, name in enumerate(CLASSES):
        for path in sorted((DATA / 'dataset-resized' / name).glob('*.jpg')):
            with Image.open(path) as im:
                rgb = im.convert('RGB')
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if digest in seen:
                if seen[digest]['label'] != label:
                    conflicts.add(digest)
                duplicates.append(str(path.relative_to(DATA)))
                continue
            row = {'path': str(path.relative_to(DATA)), 'label': label, 'sha256': digest}
            seen[digest] = row
            rows.append(row)
    conflicting_rows = [r for r in rows if r['sha256'] in conflicts]
    rows = [r for r in rows if r['sha256'] not in conflicts]
    if len(rows) < 2000:
        raise ValueError('Dataset appears incomplete')
    train, rest = train_test_split(rows, test_size=0.30, random_state=42,
                                   stratify=[r['label'] for r in rows])
    val, test = train_test_split(rest, test_size=0.5, random_state=42,
                                 stratify=[r['label'] for r in rest])
    split = {'train': train, 'val': val, 'test': test}
    save_json(DATA / 'splits.json', split)
    save_json(OUT / 'dataset.json', {
        'source': url, 'repository': 'https://github.com/garythung/trashnet',
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'seed': 42, 'duplicates_removed': duplicates,
        'conflicting_representatives_removed': conflicting_rows,
        'counts': {k: {c: sum(r['label'] == i for r in v) for i, c in enumerate(CLASSES)}
                   for k, v in split.items()}})
    print({k: len(v) for k, v in split.items()}, flush=True)


def transform(training=False):
    steps = [transforms.Resize((160, 160))]
    if training:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                  transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Images(Dataset):
    def __init__(self, rows, training=False):
        self.rows, self.tf = rows, transform(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(DATA / row['path']) as im:
            return self.tf(im.convert('RGB')), row['label']


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers, channels = [], 3
        for width in [16, 32, 64]:
            layers += [nn.Conv2d(channels, width, 3, padding=1), nn.BatchNorm2d(width),
                       nn.ReLU(), nn.MaxPool2d(2)]
            channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                        nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(128, 6))

    def forward(self, x):
        return self.classifier(self.features(x))


def build(name, pretrained=False):
    if name == 'cnn':
        return SmallCNN()
    if name == 'resnet18':
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif name == 'dnn_tc':
        # Architecture-based comparison, not a reproduction of the paper's fine-tuning.
        net = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if pretrained else None,
                                    progress=False)
    else:
        raise ValueError(f'Unknown model: {name}')
    for p in net.parameters():
        p.requires_grad = False
    net.fc = (nn.Sequential(nn.Linear(net.fc.in_features, 1024), nn.ReLU(), nn.Linear(1024, 6))
              if name == 'dnn_tc' else nn.Linear(net.fc.in_features, 6))
    return net


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(6)


def run_epoch(net, loader, device, optimizer=None, frozen=False):
    net.train(optimizer is not None)
    if frozen:
        # Keep pretrained batch-normalization statistics fixed.
        net.eval()
        net.fc.train(optimizer is not None)
    total, truth, predictions, probs = 0.0, [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = net(x)
            loss = nn.functional.cross_entropy(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += loss.item() * len(y)
        truth.extend(y.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        probs.extend(logits.detach().softmax(1).cpu().tolist())
    return {'loss': total / len(truth), 'accuracy': accuracy_score(truth, predictions),
            'macro_f1': f1_score(truth, predictions, labels=list(range(6)), average='macro', zero_division=0),
            'truth': truth, 'predictions': predictions, 'probabilities': probs}


def train(args):
    seed_all(args.seed)
    split = json.loads((DATA / 'splits.json').read_text())
    loaders = {k: DataLoader(Images(split[k], k == 'train'), batch_size=args.batch_size,
                            shuffle=k == 'train', num_workers=0) for k in ['train', 'val']}
    net = build(args.model, pretrained=True).to(args.device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, weight_decay=0.0001)
    history, best, started = [], -1.0, time.perf_counter()
    folder = OUT / args.model
    folder.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(net, loaders['train'], args.device, optimizer, args.model != 'cnn')
        va = run_epoch(net, loaders['val'], args.device)
        record = {'epoch': epoch, **{f'{phase}_{k}': float(result[k])
                  for phase, result in [('train', tr), ('val', va)] for k in ['loss', 'accuracy', 'macro_f1']}}
        history.append(record)
        print(json.dumps(record), flush=True)
        if va['macro_f1'] > best:
            best = va['macro_f1']
            torch.save({'state_dict': net.state_dict(), 'model': args.model, 'epoch': epoch,
                        'classes': CLASSES, 'image_size': 160}, folder / 'best.pt')
        save_json(folder / 'history.json', history)
    save_json(folder / 'training.json', {'seconds': time.perf_counter() - started,
        'device': args.device, 'torch': torch.__version__, 'seed': args.seed,
        'epochs': args.epochs, 'batch_size': args.batch_size, 'learning_rate': 0.001,
        'selection': 'maximum validation macro F1',
        'initialization': 'random' if args.model == 'cnn' else 'ImageNet pretrained, frozen backbone',
        'parameters': sum(p.numel() for p in net.parameters()),
        'trainable_parameters': sum(p.numel() for p in net.parameters() if p.requires_grad)})


def load_model(name):
    checkpoint = torch.load(OUT / name / 'best.pt', map_location='cpu', weights_only=True)
    net = build(name)
    net.load_state_dict(checkpoint['state_dict'])
    return net.eval(), checkpoint


def evaluate(names=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seed_all(42)
    split = json.loads((DATA / 'splits.json').read_text())
    summary_path = OUT / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in (names or MODEL_NAMES):
        net, checkpoint = load_model(name)
        result = run_epoch(net, DataLoader(Images(split['test']), batch_size=32), 'cpu')
        folder = OUT / name
        report = classification_report(result['truth'], result['predictions'], labels=list(range(6)),
                                       target_names=CLASSES, output_dict=True, zero_division=0)
        save_json(folder / 'test.json', {**result, 'report': report, 'selected_epoch': checkpoint['epoch']})
        summary[name] = {k: result[k] for k in ['accuracy', 'macro_f1']}
        matrix = confusion_matrix(result['truth'], result['predictions'], labels=list(range(6)))
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(matrix, cmap='Blues')
        ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel='Predicted', ylabel='Actual', title=f'{name}: test confusion matrix')
        plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
        for i in range(6):
            for j in range(6):
                ax.text(j, i, matrix[i, j], ha='center', va='center', color='white' if matrix[i,j] > matrix.max()/2 else 'black')
        fig.tight_layout(); fig.savefig(folder / 'confusion.png', dpi=160); plt.close(fig)
        history = json.loads((folder / 'history.json').read_text())
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for phase, color in [('train', '#176b50'), ('val', '#c58e35')]:
            ax.plot([r['epoch'] for r in history], [r[f'{phase}_loss'] for r in history],
                    label=f'{phase} loss', color=color, linewidth=2, marker='o', markersize=3)
        ax.axvline(checkpoint['epoch'], color='#6b7280', linestyle='--', label='Selected by val macro F1')
        ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title=f'{name}: training and validation loss')
        ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(folder / 'loss.png', dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for phase, color in [('train', '#176b50'), ('val', '#c58e35')]:
            ax.plot([r['epoch'] for r in history], [r[f'{phase}_accuracy'] for r in history],
                    label=f'{phase} accuracy', color=color, linewidth=2)
        ax.axvline(checkpoint['epoch'], color='#6b7280', linestyle='--', label='Selected epoch')
        ax.set(xlabel='Epoch', ylabel='Accuracy', ylim=(0, 1), title=f'{name}: accuracy')
        ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(folder / 'accuracy.png', dpi=180); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ['loss', 'accuracy']):
            for phase in ['train', 'val']:
                ax.plot([r['epoch'] for r in history], [r[f'{phase}_{metric}'] for r in history], label=phase)
            ax.set(xlabel='Epoch', ylabel=metric); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(folder / 'learning.png', dpi=160); plt.close(fig)
        with (folder / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(['path', 'actual', 'predicted', 'max_softmax'])
            for row, pred, prob in zip(split['test'], result['predictions'], result['probabilities']):
                writer.writerow([row['path'], CLASSES[row['label']], CLASSES[pred], max(prob)])
    save_json(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train', 'evaluate'])
    parser.add_argument('--model', choices=MODEL_NAMES, default='cnn')
    parser.add_argument('--models', choices=MODEL_NAMES, nargs='+', help='Models to evaluate; default all three')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    {'prepare': prepare, 'train': lambda: train(args), 'evaluate': lambda: evaluate(args.models)}[args.action]()
