"""Independent seeded 60/20/20 split; not the unpublished author split."""
import hashlib
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split
from project import DATA, OUT, CLASSES, save_json


def main():
    target = DATA / 'splits_paper_602020.json'
    if target.exists():
        raise RuntimeError('Split already exists; refusing to overwrite.')
    rows = []
    for label, name in enumerate(CLASSES):
        for path in sorted((DATA / 'dataset-resized' / name).glob('*.jpg')):
            with Image.open(path) as im:
                rgb = im.convert('RGB')
                assert min(rgb.size) >= 224, path
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            rows.append({'path': path.relative_to(DATA).as_posix(), 'label': label, 'sha256': digest})
    assert len(rows) == 2527, len(rows)
    train, rest = train_test_split(rows, test_size=0.4, random_state=42,
                                   stratify=[r['label'] for r in rows])
    val, test = train_test_split(rest, test_size=0.5, random_state=42,
                                stratify=[r['label'] for r in rest])
    split = {'train': train, 'val': val, 'test': test}
    paths = [{r['path'] for r in part} for part in split.values()]
    assert all(not paths[i] & paths[j] for i in range(3) for j in range(i))
    assert len(set.union(*paths)) == 2527
    groups = defaultdict(list)
    for phase, part in split.items():
        for row in part:
            groups[row['sha256']].append({'split': phase, **row})
    duplicates = [g for g in groups.values() if len(g) > 1]
    save_json(target, split)
    audit = {
        'seed': 42, 'total': len(rows), 'ratio': [0.6, 0.2, 0.2],
        'author_exact_split': False,
        'counts': {k: len(v) for k, v in split.items()},
        'class_counts': {k: {c: sum(r['label'] == i for r in v)
                            for i, c in enumerate(CLASSES)} for k, v in split.items()},
        'split_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'duplicate_groups_retained': duplicates,
        'cross_split_duplicate_groups': [g for g in duplicates if len({r['split'] for r in g}) > 1],
        'note': 'Original dataset retained, including conflicting duplicate labels. '
                'This is a ratio-matched experiment, not a clean held-out benchmark or exact reproduction.'}
    save_json(OUT / 'dataset_paper_602020.json', audit)
    print(audit['counts'], flush=True)
    print('Cross-split duplicate groups:', len(audit['cross_split_duplicate_groups']), flush=True)


if __name__ == '__main__':
    main()
