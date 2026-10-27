"""Frozen-backbone DNN-TC architecture comparison using two cached augmented views.

This intentionally differs from Vo et al.'s full fine-tuning protocol.
Only training/validation features are cached before checkpoint selection.
"""
import hashlib
import json
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from project import DATA, OUT, CLASSES, Images, build, seed_all, save_json, run_epoch


def main():
    seed_all(42)
    folder = OUT / 'dnn_tc'
    folder.mkdir(parents=True, exist_ok=True)
    split_bytes = (DATA / 'splits.json').read_bytes()
    split = json.loads(split_bytes)
    signature = hashlib.sha256(split_bytes).hexdigest()
    print('Loading ResNeXt101 32x8d ImageNet V1 weights...', flush=True)
    net = build('dnn_tc', pretrained=True).eval()
    head = net.fc
    net.fc = nn.Identity()
    started = time.perf_counter()
    cached, timings = {}, {}
    # Two fixed random views per training image; validation uses deterministic transforms.
    for key, phase, augmented in [('train0', 'train', True), ('train1', 'train', True), ('val', 'val', False)]:
        stamp = time.perf_counter()
        loader = DataLoader(Images(split[phase], augmented), batch_size=8, shuffle=False, num_workers=0)
        chunks, labels = [], []
        with torch.inference_mode():
            for batch, (x, y) in enumerate(loader):
                chunks.append(net(x).cpu())
                labels.append(y)
                if batch % 25 == 0:
                    print(f'{key}: {min((batch+1)*8, len(loader.dataset))}/{len(loader.dataset)} images', flush=True)
        cached[key] = (torch.cat(chunks), torch.cat(labels))
        timings[key] = time.perf_counter() - stamp
        print(f'{key} extraction: {timings[key]:.1f}s', flush=True)
    extraction_seconds = time.perf_counter() - started
    # Inference tensors become ordinary training inputs after cloning outside inference_mode.
    cached = {k: (x.clone(), y.clone()) for k, (x, y) in cached.items()}
    validation = DataLoader(TensorDataset(*cached['val']), batch_size=32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.0001)
    history, best, best_epoch, best_head = [], -1.0, None, None
    for epoch in range(1, 16):
        view = (epoch - 1) % 2
        training = DataLoader(TensorDataset(*cached[f'train{view}']), batch_size=32, shuffle=True)
        tr = run_epoch(head, training, 'cpu', optimizer)
        va = run_epoch(head, validation, 'cpu')
        record = {'epoch': epoch, 'cached_view': view, **{
            f'{phase}_{metric}': float(result[metric])
            for phase, result in [('train', tr), ('val', va)]
            for metric in ['loss', 'accuracy', 'macro_f1']}}
        history.append(record)
        if va['macro_f1'] > best:
            best, best_epoch = va['macro_f1'], epoch
            best_head = {k: v.detach().clone() for k, v in head.state_dict().items()}
        save_json(folder / 'history.json', history)
        print(json.dumps(record), flush=True)
    head.load_state_dict(best_head)
    net.fc = head
    torch.save({'state_dict': net.state_dict(), 'model': 'dnn_tc', 'epoch': best_epoch,
                'classes': CLASSES, 'image_size': 160, 'split_sha256': signature,
                'protocol': 'frozen backbone; two fixed augmented training views; 15 head epochs'}, folder / 'best.pt')
    save_json(folder / 'training.json', {
        'seconds': time.perf_counter() - started, 'feature_extraction_seconds': extraction_seconds,
        'feature_extraction_detail': timings, 'device': 'cpu', 'torch': torch.__version__,
        'seed': 42, 'epochs': 15, 'batch_size': 32, 'extraction_batch_size': 8,
        'learning_rate': 0.001, 'optimizer': 'AdamW', 'weight_decay': 0.0001,
        'selection': 'maximum validation macro F1', 'image_size': 160,
        'initialization': 'ImageNet V1 ResNeXt101 32x8d, frozen backbone',
        'parameters': sum(p.numel() for p in net.parameters()),
        'trainable_parameters': sum(p.numel() for p in net.parameters() if p.requires_grad),
        'split_sha256': signature, 'training_views': 2,
        'paper_reproduction': False,
        'source': 'https://doi.org/10.1109/ACCESS.2019.2959033',
        'differences': ['160px instead of 224px', 'frozen backbone instead of full fine-tuning',
                        'two cached augmented views instead of fresh views every epoch',
                        '15 AdamW epochs instead of paper Adam/SGD schedule',
                        'existing 70/15/15 split instead of paper 60/20/20',
                        'ReLU between FC layers is an explicit implementation choice',
                        'cross entropy on logits is equivalent to log-softmax plus NLL']})
    print('DNN-TC comparison checkpoint saved.', flush=True)


if __name__ == '__main__':
    main()
