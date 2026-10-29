# train.py CNN 訓練結果

- 架構：CNN.py，沿用上層 CNN_CIFAR.py 的六層卷積、BatchNorm、MaxPool、Dropout 與全連接層，輸出改為六類；從頭訓練。
- 輸入：RGB 32×32，Normalize(mean=0.5, std=0.5)，無隨機資料增強。
- 資料：data/dataset-resized；共 2527 張，沿用 data/splits.json 排除 6 張同圖異標籤圖片後的切分。
- 訓練 / 驗證 / 測試：1764 / 378 / 379 張；seed=42。
- 設定：5 epochs、batch size=128、Adam lr=0.001、CrossEntropyLoss、ReduceLROnPlateau(factor=0.5, patience=2)。
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，PyTorch 2.7.1+cu128。
- 依驗證 loss 選擇第 5 epoch；驗證 accuracy 46.03%。
- 保留測試集 accuracy：41.69%；macro F1：0.3376。測試集未參與訓練或最佳模型選擇。

這是沿用原始 5 epochs 設定的基準結果，準確率仍有限；並未為提高測試成績反覆調參。32×32 與原專案其他模型的 160×160 設定不同，不能視為完全相同條件的比較。

在本專案目錄、已安裝依賴的 Python 環境重現：

```bash
python train.py --data_dir data/dataset-resized --split_file data/splits.json --out_dir result/cnn_train_5epochs --epochs 5 --batch_size 128 --lr 0.001
```

best_cnn_trashnet.pth 為最佳權重；history.csv 為逐輪數據；test.json 包含每類 precision、recall、F1；predictions.csv 為逐張測試預測；loss_curve.png 與 confusion.png 為圖表。原始訓練程式備份於 train_original.py。
