"""Check split integrity and model serialization without training on test data."""
import json
import tempfile
from pathlib import Path
import torch
from project import DATA, CLASSES, Images, build, seed_all

seed_all(42)
split = json.loads((DATA / 'splits.json').read_text())
assert set(split) == {'train', 'val', 'test'}
hashes = {k: {r['sha256'] for r in v} for k, v in split.items()}
for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
    assert hashes[a].isdisjoint(hashes[b]), f'Leakage: {a}/{b}'
for name, rows in split.items():
    assert set(r['label'] for r in rows) == set(range(6))
    assert len(hashes[name]) == len(rows)
x, y = Images(split['train'])[0]
assert x.shape == (3, 160, 160) and torch.isfinite(x).all()
for name in ['cnn', 'resnet18']:
    net = build(name).eval()
    result = net(x.unsqueeze(0))
    assert result.shape == (1, 6) and torch.isfinite(result).all()
    if name == 'resnet18':
        assert all(n.startswith('fc.') for n, p in net.named_parameters() if p.requires_grad)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'weights.pt'
        torch.save(net.state_dict(), path)
        loaded = build(name).eval()
        loaded.load_state_dict(torch.load(path, weights_only=True))
        assert torch.allclose(result, loaded(x.unsqueeze(0)))
print('PASS: distinct split hashes, six classes, preprocessing, forward pass, checkpoint round trip')
