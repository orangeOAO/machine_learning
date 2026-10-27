# RTX 5060 專用環境

保留 `.venv` 的課程PyTorch 2.2環境，另建立WSL Python 3.10環境 `/home/orange/.venvs/trashnet-gpu`。

選用官方PyTorch 2.7.1 + TorchVision 0.22.1，CUDA 12.8 wheels。官方2.7版本公告列出Blackwell支援： https://pytorch.org/blog/pytorch-2-7/

安裝（從專案根目錄，在WSL執行）：

```bash
mkdir -p /home/orange/.cache/trashnet-pip-tmp
export TMPDIR=/home/orange/.cache/trashnet-pip-tmp
.venv/bin/python -m venv /home/orange/.venvs/trashnet-gpu
/home/orange/.venvs/trashnet-gpu/bin/python -m pip install --upgrade pip
/home/orange/.venvs/trashnet-gpu/bin/python -m pip install https://download.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl https://download.pytorch.org/whl/cu128/torchvision-0.22.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl --index-url https://pypi.org/simple
/home/orange/.venvs/trashnet-gpu/bin/python -m pip install -r midterm/requirements-gpu.txt
/home/orange/.venvs/trashnet-gpu/bin/python midterm/check_gpu.py
```

`check_gpu.py`實際執行224×224、batch 8、全模型反向傳播與Adam更新，成功才寫入`outputs/gpu_check.json`；CUDA可用旗標本身不代表模型能訓練。

GPU論文訓練流程版獨立存於`outputs/dnn_tc_paper_gpu/`：

```bash
/home/orange/.venvs/trashnet-gpu/bin/python -u midterm/train_dnntc_paper.py --device cuda --output-name dnn_tc_paper_gpu
# 若已存在last.pt，使用：
/home/orange/.venvs/trashnet-gpu/bin/python -u midterm/train_dnntc_paper.py --device cuda --output-name dnn_tc_paper_gpu --resume
```

輸入、batch、110輪Adam/SGD流程不因GPU改變。實際torch與CUDA版本記錄於training.json。CPU與GPU時間不可直接比較為架構快慢。



## 2026-10-25 實測

已完成安裝與 `pip check`，沒有相依性衝突。實測結果見 `outputs/gpu_check.json`：

- PyTorch 2.7.1+cu128、CUDA 12.8、TorchVision 0.22.1+cu128。
- NVIDIA GeForce RTX 5060 Laptop GPU，compute capability 12.0。
- 224×224、batch 8，全模型 forward/backward/Adam 三步通過。
- 峰值已配置顯存 3.01 GiB（不包含其他程序或 CUDA 額外保留空間）。
- 三步耗時約 3.87、4.34、0.32 秒；短測含暖機，不能直接當完整 epoch 時間。

GPU 正式訓練另開新實驗，保留 CPU 第 1 輪檢查點。執行紀錄見 `outputs/dnn_tc_paper_gpu/run.log`，錯誤見 `error.log`。`status.json` 是最後寫入狀態，程序中斷後不會自動改成停止，需搭配程序與 log 確認。

安裝時遇到 NVIDIA 依賴下載中斷，因此上方命令使用官方 torch/torchvision wheel 及 PyPI 依賴。WSL 的 `/tmp` 是容量有限的 tmpfs，使用 Linux 磁碟上的 TMPDIR 避免安裝暫存耗盡。
