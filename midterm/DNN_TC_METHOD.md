# DNN-TC 架構比較版

參考：Vo et al., *A Novel Framework for Trash Classification Using Deep Transfer Learning*, IEEE Access, 2019. DOI: https://doi.org/10.1109/ACCESS.2019.2959033

使用 ResNeXt-101 骨幹，在 global average pooling 後使用輸出 1024 與 6 維的兩個全連接層。本實作選擇 TorchVision `resnext101_32x8d` 的 ImageNet V1 權重，兩全連接層之間加 ReLU。這些實作選擇需與核心架構一起揭露，不能宣稱逐項完全復現作者程式。

## 本次比較設定

三模型共用同一份 `data/splits.json`。CNN 和 ResNet18 使用原有已完成結果，DNN-TC 使用相同的驗證集及 379 張測試圖片。全部選用 validation macro F1 最佳 checkpoint。

| 項目 | CNN | ResNet18 | DNN-TC 架構比較版 |
|---|---|---|---|
| 輸入 | 160×160 | 160×160 | 160×160 |
| 初始化 | 隨機 | ImageNet 預訓練 | ImageNet V1 預訓練 |
| 更新參數 | 全模型 | 最後分類層 | 兩個全連接層 |
| 增強 | 每輪重新隨機 | 每輪重新隨機 | 兩組固定增強視圖交替 |
| 訓練輪數 | 15 | 15 | 15 |
| Optimizer | AdamW | AdamW | AdamW |
| lr / weight decay | 0.001 / 0.0001 | 同左 | 同左 |
| 訓練 batch size | 32 | 32 | 分類頭 32，特徵擷取 8 |

骨幹在 eval 模式且完全凍結，因此對固定圖片的特徵可重用。快取只包含訓練與驗證資料，測試集於選定 checkpoint 後才推論。此方法降低重複計算，但兩組固定增強與每輪重新隨機增強不同，不能視為完全相同的訓練預算。

程式輸出 logits，以 CrossEntropyLoss 訓練，數學上相當於 log-softmax 接 NLLLoss。Demo 才以 softmax 轉成分數。

## 與論文的差異

論文描述 224×224、60/20/20 切分、Adam 後接 SGD 的微調流程；本實作保留現有專案切分與凍結骨幹，使用 15 輪 AdamW。論文的「前 10 輪加後 100 輪」及「共 100 輪」文字有不一致，本專案不宣稱重現其完整訓練時程。

論文報告的 TrashNet 94% 是文獻結果，不是本專案的成績。正式成績只引用 `outputs/dnn_tc/test.json`。

## 執行

```bash
python midterm/train_dnntc.py
python midterm/project.py evaluate --models dnn_tc
python midterm/make_assets.py
python midterm/write_report.py
python midterm/check_demo.py
python midterm/demo.py
```

所有模型以 CPU 執行。訓練時間不含權重下載和最後測試，DNN-TC 的計時包含特徵擷取。時間受資料增強方式、快取、batch size 與系統負載影響，不是單純架構速度排名。
