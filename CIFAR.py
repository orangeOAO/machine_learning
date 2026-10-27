
import os
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import random

import matplotlib.pyplot as plt
from matplotlib import font_manager

from pathlib import Path
from CNN_CIFAR import CNN


CLASS_NAMES =[
    "airplane", "car", "bird", "cat", "deer", "dog", 
    "frog", "horse", "boat", "truck"
]
DATA_DIR = "data"
OUT_DIR = "result"
CKPT_PATH = Path(OUT_DIR) / "best_cnn_for_CIFAR.pth"
ROWS, COLS = 2, 5
SEED = 42
MEAN, STD = 0.5, 0.5
FIGSIZE = (16, 8)
TITLE_FZ = 11
ZH_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\msjhl.ttc",
    r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

def denorm_img(x):
    x = x.detach().cpu()
    x = x * STD + MEAN
    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0)
    return x.numpy()



def set_zh_font():
    from pathlib import Path as _Path
    for p in ZH_FONT_CANDIDATES:
        if _Path(p).exists():
            prop = font_manager.FontProperties(fname=p)
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[INFO] 使用中文字型：{prop.get_name()} ({p})")
    print("[WARN] not found chinese font, plz install 微軟正黑體/Noto Sans CJK")


def set_seed(seed:int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(SEED)
    set_zh_font()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=tfm)
    model = CNN(num_classes=10).to(device)
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        print(f"[INFO] loaded checkpoint: {CKPT_PATH}")
    else:
        print(f"[WARN] checkpoint not found: {CKPT_PATH}")
    
    model.eval()

    N = ROWS * COLS
    indices = list(range(len(test_set)))
    random.shuffle(indices)
    indices = indices[:N]

    #推論
    images = []
    gts = []
    preds = []
    confs = []

    with torch.no_grad():
        for idx in indices:
            img_t, label = test_set[idx]
            logits = model(img_t.unsqueeze(0).to(device))
            prob = F.softmax(logits, dim=1)[0]
            pred_id = int(torch.argmax(prob).item())
            conf = float(prob[pred_id].item())

            images.append(img_t.cpu())
            gts.append(label)
            preds.append(pred_id)
            confs.append(conf)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)

    for i in range(N):
        ax = fig.add_subplot(ROWS, COLS, i+1)
        img = denorm_img(images[i])
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.axis("off")

        gt_name = CLASS_NAMES[gts[i]]
        pred_name = CLASS_NAMES[preds[i]]
        conf_txt = f"{confs[i]:.3f}"

        title = f"real:{gt_name}\npred:{pred_name}\nconf:{conf_txt}"
        ax.set_title(title, color="green", fontsize=TITLE_FZ, pad=6)
    fig.suptitle("result", fontsize=16, y=0.98)

    out_path = Path(OUT_DIR) / "test_grid.png"
    fig.savefig(out_path, dpi = 180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] output:{out_path.resolve()}")


if __name__ == '__main__':
    main()
    