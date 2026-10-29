
import os
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import random
import json
import csv
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import matplotlib.pyplot as plt
from matplotlib import font_manager

from pathlib import Path
from CNN import CNN

FEATURE = "batch128_AdamW_Dropout"
CLASS_NAMES =[
    "cardboard", "glass", "metal", "paper", "plastic", "trash"
]
DATA_DIR = "data/dataset-resized"
OUT_DIR = "result"
ROWS, COLS = 2, 5
SEED = 42
MEAN, STD = 0.2861, 0.3530
FIGSIZE = (16, 8)
TITLE_FZ = 11
ZH_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\msjhl.ttc",
    r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]
def set_zh_font():
    from pathlib import Path as _Path
    for p in ZH_FONT_CANDIDATES:
        if _Path(p).exists():
            prop = font_manager.FontProperties(fname=p)
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[INFO] 使用中文字型：{prop.get_name()} ({p})")
            return
    print("[WARN] not found chinese font, plz install 微軟正黑體/Noto Sans CJK")


def set_seed(seed:int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_loss(csv_path: Path, out_path: Path):
    epochs, tr_losses, val_losses, tr_accs, val_accs = [], [], [], [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            e, tl, vl, ta, va = line.strip().split(',')
            epochs.append(int(e))
            tr_losses.append(float(tl))
            val_losses.append(float(vl))
            tr_accs.append(float(ta))
            val_accs.append(float(va))
    plt.figure(figsize=(6,4))
    plt.plot(epochs, tr_losses, label='train_loss')
    plt.plot(epochs, val_losses, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    set_seed(SEED)
    set_zh_font()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--split_file', type=str, default='data/splits.json')
    parser.add_argument('--out_dir', type=str, default='result/resnet101_25epochs')
    parser.add_argument('--feature', type=str, default='batch128_AdamW_Dropout')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model', choices=['resnet101'], default='resnet101')
    args = parser.parse_args()

    if not Path(args.data_dir).is_dir():
        raise FileNotFoundError(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"data_dir={args.data_dir}, out_dir={args.out_dir}")

    # model = nn.Sequential(
    #         nn.Conv2d(1, 16, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2),
    #         nn.Conv2d(16, 32, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2),
    #         nn.Flatten(),
    #         nn.Linear(32 * 7 * 7,10),
    #     )
    # print(sum(p.numel() for p in model.parameters()))

    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    train_tfm = tfm
    if args.model == 'resnet101':
        tfm = models.ResNet101_Weights.IMAGENET1K_V2.transforms()
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    full_train = datasets.ImageFolder(args.data_dir, transform=tfm)
    augmented = datasets.ImageFolder(args.data_dir, transform=train_tfm)
    assert full_train.classes == CLASS_NAMES, full_train.classes
    splits = json.loads(Path(args.split_file).read_text(encoding='utf-8'))
    lookup = {Path(p).resolve(): (i, label) for i, (p, label) in enumerate(full_train.samples)}
    subsets, seen_paths, seen_hashes = {}, set(), set()
    for name in ('train', 'val', 'test'):
        indices = []
        for row in splits[name]:
            path = (Path(args.data_dir) / Path(row['path']).relative_to('dataset-resized')).resolve()
            index, label = lookup[path]
            assert label == row['label']
            assert path not in seen_paths and row['sha256'] not in seen_hashes, 'Split overlap'
            seen_paths.add(path)
            seen_hashes.add(row['sha256'])
            indices.append(index)
        subsets[name] = torch.utils.data.Subset(augmented if name == 'train' else full_train, indices)
    train_set, val_set, test_set = [subsets[k] for k in ('train', 'val', 'test')]
    print('Split sizes:', {k: len(v) for k, v in subsets.items()})
    Path(args.out_dir, 'splits.json').write_text(json.dumps(splits, indent=2), encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"DataLoader is ready (batch_size={args.batch_size})")

    #=====================training======================

    if args.model == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(model.fc.in_features, len(CLASS_NAMES)),
        )
        model = model.to(device)
    else:
        model = CNN(num_classes=len(CLASS_NAMES)).to(device)
    print(f"using device:{device}")
    criterion = nn.CrossEntropyLoss()
    parameters = model.parameters()
    if args.model == 'resnet101':
        parameters = [
            {'params': [p for name, p in model.named_parameters() if not name.startswith('fc.')], 'lr': args.lr * 0.1},
            {'params': model.fc.parameters(), 'lr': args.lr},
        ]
    optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and args.model == 'resnet101')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    print("[S5] Loss/Optimizer is ready")

    loss_csv = os.path.join(args.out_dir, 'history.csv')
    with open(loss_csv, 'w', encoding='utf-8') as f:
        f.write('epoch,train_loss,val_loss,train_acc,val_acc\n')

    best_val_loss =float('inf')
    best_path = os.path.join(args.out_dir, f'best_{args.model}_{args.feature}.pth')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_acc += (logits.argmax(1) == labels).float().sum().item()
            n += bs
        train_loss = running_loss / n
        train_acc = running_acc / n

        model.eval()
        val_loss, val_acc, m = 0.0, 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                bs = labels.size(0)
                val_loss += loss.item() * bs
                val_acc += (logits.argmax(1) == labels).float().sum().item()
                m += bs
        val_loss /= m
        val_acc /= m

        scheduler.step(val_loss)
        
        with open(loss_csv, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_acc:.4f},{val_acc:.4f}\n")
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        # dt = time.time() - t0
        # print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} ({dt:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch':epoch,
                'model_state':model.state_dict(),
                'optimizer_state':optimizer.state_dict(),
                'val_loss':val_loss,
                'val_acc':val_acc,
                'classes':CLASS_NAMES,
                'image_size':224 if args.model == 'resnet101' else 32,
                'model':args.model,
                'pretrained_weights':'IMAGENET1K_V2' if args.model == 'resnet101' else None,
            }, best_path)
            print(f" -> Saved new best to {best_path}")
        
 
    print(f"Training complete: {loss_csv}")
    plot_loss(loss_csv, os.path.join(args.out_dir, 'loss_curve.png'))
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    targets, predictions = [], []
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            test_loss += criterion(logits, labels.to(device)).item() * len(labels)
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'macro_f1': f1_score(targets, predictions, average='macro', zero_division=0),
        'loss': test_loss / len(test_set),
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': checkpoint['val_loss'],
        'best_val_accuracy': checkpoint['val_acc'],
        'classification_report': classification_report(targets, predictions, labels=list(range(6)), target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(targets, predictions, labels=list(range(6))).tolist(),
        'config': vars(args),
        'seed': SEED,
        'split_sizes': {k: len(v) for k, v in subsets.items()},
        'device': str(device),
        'input_size': [224, 224] if args.model == 'resnet101' else [32, 32],
    }
    Path(args.out_dir, 'test.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    with open(Path(args.out_dir, 'predictions.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'true_class', 'predicted_class'])
        for row, target, prediction in zip(splits['test'], targets, predictions):
            writer.writerow([row['path'], CLASS_NAMES[target], CLASS_NAMES[prediction]])
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = metrics['confusion_matrix']
    ax.imshow(cm, cmap='Blues')
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel='Predicted', ylabel='True', title='Test confusion matrix')
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(Path(args.out_dir, 'confusion.png'))
    plt.close(fig)
    print(f"Test accuracy={metrics['accuracy']:.4f}, macro F1={metrics['macro_f1']:.4f}, best epoch={checkpoint['epoch']}")
if __name__ == '__main__':
    main()
    

