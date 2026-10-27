"""Verify real full-model CUDA forward/backward/Adam, not just availability."""
import json
import time
import torch
from project import build, save_json, OUT

assert torch.cuda.is_available(), 'CUDA unavailable'
torch.set_num_threads(6)
net = build('dnn_tc').cuda()
for p in net.parameters():
    p.requires_grad = True
net.train()
optimizer = torch.optim.Adam(net.parameters(), lr=.001)
x = torch.rand(8, 3, 224, 224, device='cuda')
y = torch.arange(8, device='cuda') % 6
times = []
for step in range(3):
    torch.cuda.synchronize()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(net(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
result = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
          'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
          'batch_size': 8, 'image_size': 224, 'step_seconds': times,
          'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
          'full_model_adam_passed': True}
save_json(OUT / 'gpu_check.json', result)
print(json.dumps(result, indent=2), flush=True)
