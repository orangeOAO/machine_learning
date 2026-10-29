# CNN 從頭訓練與 ResNet101 微調

> 2026-10-28 更新：train_cnn.py 與 train_resnet101.py 的預設已改為 25 epochs，輸出分別為 result/cnn_25epochs 與 result/resnet101_25epochs。以下 5／10 輪成績保留為先前實驗紀錄；最新比較以 result/comparison_25epochs/summary.json 與 25 輪簡報為準。

Fine-tuning（微調）也是 training。這次比較的是「從隨機權重開始訓練」與「從預訓練權重繼續訓練」。ResNet101 本身也是卷積神經網路（CNN）。

兩者都經過：輸入圖片 → forward 預測 → loss 計算誤差 → backward 求梯度 → optimizer 更新參數。差別主要在參數的起點，以及哪些參數更新、使用多大的學習率。

## 程式與結果保留

- train_cnn.py：獨立保存目前訓練流程，固定選用自訂 CNN；預設 5 epochs、batch size 128、lr=0.001。
- CNN.py：自訂 CNN 架構，六層卷積及全連接分類器。
- train_resnet101.py：獨立保存目前訓練流程，固定選用 ResNet101；預設 10 epochs、batch size 8、分類頭 lr=0.0001、骨幹 lr=0.00001。
- 兩個訓練檔不依賴 train.py，但保留相同流程的條件分支以方便對照；--model 限定為各自模型。
- train.py：原本可選模型的入口仍保留。
- train_original.py：最早收到的 CIFAR-10 程式備份，並不是這次垃圾資料集實驗版本。
- result/cnn_train_5epochs：已完成的 CNN 權重、曲線、測試結果。
- result/resnet101_finetune：已完成的 ResNet101 權重、曲線、測試結果。

在本專案目錄、既有 PyTorch 環境執行：

```bash
python train_cnn.py
python train_resnet101.py
```

預設新結果分別寫到 result/cnn_train_rerun 與 result/resnet101_finetune_rerun，因此不會覆蓋上述已完成實驗。再次執行會覆蓋 rerun 目錄的同名結果；不同實驗請使用 --out_dir 指定新目錄。這次僅保存與檢查程式，沒有重新訓練。

## 用本次實驗理解

自訂 CNN 的卷積與全連接權重一開始為隨機初始化，需要從 1764 張訓練圖學習特徵與分類。ResNet101 先載入 ImageNet 預訓練權重，再將原本 1000 類的分類層換成隨機初始化的 Linear(2048, 6)，然後用垃圾圖片調整整個模型。

預訓練骨幹已有可轉用的影像特徵，所以此次以較小學習率更新；新分類層需要學習六個垃圾類別，使用較大學習率。微調不保證一定優於從頭訓練，仍須依驗證及測試結果判斷。

若將骨幹 requires_grad 設為 False 並固定其 BatchNorm 統計，只訓練分類頭，則是使用固定特徵抽取器的遷移學習。本次 ResNet101 沒有凍結骨幹，包含 BatchNorm 都參與訓練，屬於全模型微調。

## 實際設定與成績

| 項目 | 自訂 CNN | ResNet101 |
|---|---|---|
| 起始權重 | 隨機初始化 | ImageNet V2 預訓練，分類頭隨機初始化 |
| 模型輸入 | 32×32 | 224×224 |
| 更新範圍 | 全模型 | 全模型 |
| Epochs | 5 | 10 |
| Batch size | 128 | 8 |
| 初始學習率 | 0.001 | 骨幹 0.00001、分類頭 0.0001 |
| 隨機增強 | 無 | 隨機裁切縮放、水平翻轉 |
| 最佳驗證 accuracy | 46.03% | 90.74% |
| 測試 accuracy | 41.69% | 89.97% |
| 測試 macro F1 | 0.3376 | 0.8694 |

兩者使用同一組 1764/378/379 的 train/validation/test 切分與 seed=42，以 validation loss 選擇最佳模型，最後評估 test。訓練集更新參數；驗證集選模型；測試集提供最後的評估。不可根據測試成績反覆調參。

## 報告可以怎麼說

「本實驗比較自訂 CNN 從頭訓練與預訓練 ResNet101 全模型微調兩種設定。在相同資料切分下，後者的測試準確率由 41.69% 提高至 89.97%。但兩組的架構、輸入解析度、增強、訓練輪數及 batch size 也不同，因此不能將全部差距歸因於預訓練。」

若要單獨驗證預訓練的效果，可固定 ResNet101 架構與其他條件，對照 weights=None 與 ImageNet 權重。這是後續實驗設計，尚未執行。

參考：https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
