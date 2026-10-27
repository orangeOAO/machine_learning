# DNN-TC 論文訓練流程版

這是新增實驗，原三模型結果保留在原資料夾。新結果存於 `outputs/dnn_tc_paper/`。

- 224×224 隨機裁切與水平翻轉，RGB縮放至0–1。
- 全模型微調，包含骨幹與BatchNorm；不快取特徵。
- batch size 8。
- 第1–10輪 Adam，lr=0.001，betas=(0.9,0.999)。
- 第11–110輪 SGD，lr=0.0001，momentum=0、weight_decay=0。
- 依驗證Accuracy選最佳checkpoint，110輪結束後才評估測試集。
- 每輪保存last.pt（模型、optimizer、隨機狀態）與best.pt。

## 明確假設與差異

論文同時描述「前10輪＋後100輪」及「共100輪」，此處採110輪解讀。未充分交代的解凍範圍、SGD momentum、權重版本及分類頭ReLU均為明確實作選擇。採TorchVision ResNeXt101 32×8d ImageNet V1。

為比較現有三模型，沿用去除衝突重複圖片後的70/15/15切分，不改成論文60/20/20。驗證與測試採固定中心裁切，避免隨機評估。故稱「論文訓練流程版」，不能宣稱完全復現或保證達94%。

## 執行與查看

在WSL的midterm目錄：

```bash
../.venv/bin/python -u train_dnntc_paper.py
# 中斷後從上一個完整epoch繼續
../.venv/bin/python -u train_dnntc_paper.py --resume
```

`outputs/dnn_tc_paper/status.json`標示當前輪數，`history.json`記錄已完成的輪數；訓練訊息亦輸出於啟動終端。請勿同時啟動第二個寫入同資料夾的訓練程序。

110輪全部完成後才會產生 `test.json`、`loss.png`、`accuracy.png`、`confusion.json`。未完成前不應把中途驗證成績當成最終測試成績。

CPU訓練需要電腦保持開機且WSL持續執行；睡眠會暫停計算。若關機或程序中斷，可用上述resume命令續跑。
