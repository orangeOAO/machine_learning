"""Paper-aligned training schedule; disclosed assumptions, same comparison split.

Run with --resume to continue an interrupted run. Never reads test data until
all 110 epochs finish. Existing three-model checkpoints are not modified.
"""
import argparse
import hashlib
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from project import DATA, OUT, CLASSES, Images, build, seed_all, save_json, run_epoch


def dataset(rows, training):
    ds = Images(rows)
    # Source images are larger than 224. Deterministic validation/test prevents
    # checkpoint selection being affected by random evaluation crops.
    ds.tf = transforms.Compose(
        ([transforms.RandomHorizontalFlip(), transforms.RandomCrop(224)] if training
         else [transforms.CenterCrop(224)]) + [transforms.ToTensor()])
    return ds


def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-name', choices=['dnn_tc_paper', 'dnn_tc_paper_gpu', 'dnn_tc_paper_602020'], default='dnn_tc_paper')
    args = parser.parse_args()
    seed_all(42)
    folder = OUT / args.output_name
    folder.mkdir(parents=True, exist_ok=True)
    paper_split = args.output_name == 'dnn_tc_paper_602020'
    split_name = 'splits_paper_602020.json' if paper_split else 'splits.json'
    split_bytes = (DATA / split_name).read_bytes()
    signature = hashlib.sha256(split_bytes).hexdigest()
    split = json.loads(split_bytes)
    latest = folder / 'last.pt'
    if latest.exists() and not args.resume:
        raise RuntimeError('Existing run found. Use --resume; refusing to overwrite.')
    net = build('dnn_tc', pretrained=not args.resume).to(args.device)
    for p in net.parameters():
        p.requires_grad = True
    loaders = {k: DataLoader(dataset(split[k], k == 'train'), batch_size=8,
                            shuffle=k == 'train', num_workers=0) for k in ['train', 'val']}
    history, best, elapsed, start_epoch = [], -1.0, 0.0, 1
    state = None
    if args.resume:
        # Own locally generated checkpoint includes RNG states for exact continuation.
        state = torch.load(latest, map_location=args.device, weights_only=False)
        if state['split_sha256'] != signature:
            raise ValueError('Split changed since checkpoint')
        net.load_state_dict(state['state_dict'])
        history, best, elapsed = state['history'], state['best'], state['seconds']
        start_epoch = state['epoch'] + 1
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if args.device.startswith('cuda') and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
    metadata = {
        'status': 'training', 'epochs': 110, 'batch_size': 8, 'image_size': 224,
        'device': args.device, 'seed': 42, 'torch': str(torch.__version__),
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        'schedule': 'Adam lr=0.001 betas=(0.9,0.999), epochs 1-10; SGD lr=0.0001, epochs 11-110',
        'weight_decay': 0, 'sgd_momentum': 0, 'selection': 'maximum validation accuracy',
        'trainable_parameters': sum(p.numel() for p in net.parameters()),
        'split_sha256': signature, 'split_file': split_name,
        'split_counts': {k: len(v) for k, v in split.items()},
        'source': 'https://doi.org/10.1109/ACCESS.2019.2959033',
        'paper_reproduction': False,
        'assumptions_and_differences': [
            '110 epochs follows first 10 plus next 100; paper also inconsistently says 100 total',
            'All layers including batch normalization fine-tuned; exact freeze policy unspecified',
            'TorchVision ResNeXt101 32x8d ImageNet V1 and ReLU head are explicit choices',
            ('Original 2527 images, stratified 60/20/20 seed 42; author split unavailable; duplicates retained'
             if paper_split else 'Same cleaned 70/15/15 split as three-model comparison, not paper 60/20/20'),
            'Train random 224 crop and horizontal flip; deterministic center crop for validation/test',
            'ToTensor scales RGB to 0-1; no ImageNet mean/std normalization',
            'SGD momentum and weight decay set to zero because not specified']}
    save_json(folder / 'training.json', metadata)
    optimizer = None
    started = time.perf_counter()
    for epoch in range(start_epoch, 111):
        if optimizer is None or epoch == 11:
            optimizer = (torch.optim.Adam(net.parameters(), lr=.001, betas=(.9, .999))
                         if epoch <= 10 else torch.optim.SGD(net.parameters(), lr=.0001))
            if state is not None and epoch == start_epoch and epoch != 11:
                optimizer.load_state_dict(state['optimizer'])
            state = None
        stamp = time.perf_counter()
        save_json(folder / 'status.json', {'status': 'training', 'epoch_in_progress': epoch,
                                          'completed_epochs': epoch - 1, 'total_epochs': 110})
        print(f'Epoch {epoch}/110: {type(optimizer).__name__}, full backbone training', flush=True)
        tr = run_epoch(net, loaders['train'], args.device, optimizer)
        va = run_epoch(net, loaders['val'], args.device)
        record = {'epoch': epoch, 'optimizer': type(optimizer).__name__,
                  'seconds': time.perf_counter() - stamp,
                  **{f'{phase}_{metric}': float(result[metric])
                     for phase, result in [('train', tr), ('val', va)]
                     for metric in ['loss', 'accuracy', 'macro_f1']}}
        history.append(record)
        if va['accuracy'] > best:
            best = va['accuracy']
            atomic_save({'state_dict': net.state_dict(), 'epoch': epoch, 'classes': CLASSES,
                         'image_size': 224, 'split_sha256': signature}, folder / 'best.pt')
        seconds = elapsed + time.perf_counter() - started
        atomic_save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                     'epoch': epoch, 'history': history, 'best': best, 'seconds': seconds,
                     'split_sha256': signature, 'python_rng': random.getstate(),
                     'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else None}, latest)
        save_json(folder / 'history.json', history)
        save_json(folder / 'training.json', {**metadata, 'completed_epochs': epoch, 'seconds': seconds})
        print(json.dumps(record), flush=True)
    checkpoint = torch.load(folder / 'best.pt', map_location=args.device, weights_only=True)
    net.load_state_dict(checkpoint['state_dict'])
    result = run_epoch(net, DataLoader(dataset(split['test'], False), batch_size=8), args.device)
    result['report'] = classification_report(result['truth'], result['predictions'], labels=list(range(6)),
                                               target_names=CLASSES, output_dict=True, zero_division=0)
    result['selected_epoch'] = checkpoint['epoch']
    save_json(folder / 'test.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ['loss', 'accuracy']:
        fig, ax = plt.subplots(figsize=(8, 5))
        for phase in ['train', 'val']:
            ax.plot([r['epoch'] for r in history], [r[f'{phase}_{metric}'] for r in history], label=phase)
        ax.axvline(10.5, linestyle='--', color='grey', label='Adam -> SGD')
        ax.set(xlabel='Epoch', ylabel=metric, title='DNN-TC paper-aligned schedule')
        ax.legend(); fig.tight_layout(); fig.savefig(folder / f'{metric}.png', dpi=180); plt.close(fig)
    matrix = confusion_matrix(result['truth'], result['predictions'], labels=list(range(6)))
    save_json(folder / 'confusion.json', matrix.tolist())
    save_json(folder / 'training.json', {**metadata, 'status': 'complete', 'completed_epochs': 110,
                                        'seconds': elapsed + time.perf_counter() - started})
    save_json(folder / 'status.json', {'status': 'complete', 'completed_epochs': 110,
                                      'accuracy': result['accuracy'], 'macro_f1': result['macro_f1']})
    print('Completed training and held-out test evaluation.', flush=True)


if __name__ == '__main__':
    main()
