"""Create slide images and a reproducible demo shortlist from actual outputs."""
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from project import ROOT, DATA, OUT, CLASSES

assets = ROOT / 'assets'
assets.mkdir(exist_ok=True)
split = json.loads((DATA / 'splits.json').read_text())
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
for label, ax in enumerate(axes.flat):
    row = next(r for r in split['train'] if r['label'] == label)
    with Image.open(DATA / row['path']) as im:
        ax.imshow(im)
    ax.set_title(CLASSES[label], fontsize=16); ax.axis('off')
fig.tight_layout(); fig.savefig(assets / 'examples.png', dpi=160); plt.close(fig)
if (OUT / 'summary.json').exists():
    samples = ROOT / 'demo_samples'
    samples.mkdir(exist_ok=True)
    results = {n: json.loads((OUT / n / 'test.json').read_text()) for n in ['cnn', 'resnet18', 'dnn_tc']}
    chosen = []
    conditions = [
        ('all_correct', lambda i, y: all(r['predictions'][i] == y for r in results.values())),
        ('resnet_only_correct', lambda i, y: results['cnn']['predictions'][i] != y and results['resnet18']['predictions'][i] == y),
        ('resnet_wrong', lambda i, y: results['resnet18']['predictions'][i] != y),
        ('dnntc_correct_others_wrong', lambda i, y: results['dnn_tc']['predictions'][i] == y and
         any(results[n]['predictions'][i] != y for n in ['cnn', 'resnet18'])),
        ('dnntc_wrong', lambda i, y: results['dnn_tc']['predictions'][i] != y)]
    for kind, condition in conditions:
        for i, row in enumerate(split['test']):
            if condition(i, row['label']):
                source = DATA / row['path']
                target = samples / f'{kind}_{CLASSES[row["label"]]}_{source.name}'
                shutil.copyfile(source, target)
                chosen.append({'kind': kind, 'file': target.name, 'actual': CLASSES[row['label']],
                               **{n: CLASSES[r['predictions'][i]] for n, r in results.items()}})
                break
    (samples / 'answers.json').write_text(json.dumps(chosen, indent=2, ensure_ascii=False), encoding='utf-8')
    for name, result in results.items():
        history = json.loads((OUT / name / 'history.json').read_text())
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for phase, color in [('train', '#176b50'), ('val', '#c58e35')]:
            ax.plot([r['epoch'] for r in history], [r[f'{phase}_accuracy'] for r in history],
                    label=f'{phase} accuracy', color=color, linewidth=2)
        ax.axvline(result['selected_epoch'], linestyle='--', color='#6b7280', label='Selected epoch')
        ax.set(xlabel='Epoch', ylabel='Accuracy', ylim=(0, 1), title=f'{name}: accuracy')
        ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(OUT / name / 'accuracy.png', dpi=180); plt.close(fig)
        wrong = [i for i, (a, b) in enumerate(zip(result['truth'], result['predictions'])) if a != b][:6]
        if not wrong:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        for ax in axes.flat:
            ax.axis('off')
        for ax, i in zip(axes.flat, wrong):
            with Image.open(DATA / split['test'][i]['path']) as im:
                ax.imshow(im)
            ax.set_title(f'True: {CLASSES[result["truth"][i]]}\nPred: {CLASSES[result["predictions"][i]]}')
        fig.tight_layout(); fig.savefig(assets / f'{name}_errors.png', dpi=160); plt.close(fig)
print('Assets ready:', assets)
