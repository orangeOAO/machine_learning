
import os
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import time
import random

import matplotlib.pyplot as plt
from matplotlib import font_manager

from pathlib import Path
from CNN_template import CNN


CLASS_NAMES =[
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
DATA_DIR = "data"
OUT_DIR = "result"
CKPT_PATH = Path(OUT_DIR) / "best_cnn.pth"
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
            prop = font_manager.FontProperties(frame=p)
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[INFO] 使用中文字型：{prop.get_name()} ({p})")
    print("[WARN] not found chinese font, plz install 微軟正黑體/Noto Sans CJK")

def denorm_img(x):
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    x = x * STD + MEAN
    return x.clamp(0, 1)

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
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='result')
    parser.add_argument('--epochs', type=int, default='5')
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"data_dir={args.data_dir}, out_dir={args.out_dir}")


    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])


    full_train = datasets.FashionMNIST(
            root=args.data_dir, train=True, download=True, transform=tfm
    )
    val_len = 10000
    train_len = len(full_train) - val_len
    train_set, val_set = random_split(full_train, [train_len, val_len])
    device = torch.device('cpu')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"DataLoader is ready (batch_size={args.batch_size})")

    #=====================training======================

    model = CNN(num_classes=10).to(device)
    print(f"using device:{device}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    print("[S5] Loss/Optimizer is ready")

    loss_csv = os.path.join(args.out_dir, 'loss_csv')
    with open(loss_csv, 'w', encoding='utf-8') as f:
        f.write('epoch train_loss\n')

    best_val_loss =float('inf')
    best_path = os.path.join(args.out_dir, 'best_cnn.pth')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

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
            }, best_path)
            print(f" -> Saved new best to {best_path}")
        
    print("The traing is complete, loss have already been wrote in {loss_csv}")
    plot_loss(loss_csv, os.path.join(args.out_dir, 'loss_curve.png'))
if __name__ == '__main__':
    main()
    