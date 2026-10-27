# PyTorch 期中專案：垃圾圖片分類

研究問題：在小型垃圾圖片資料集上，ImageNet 遷移學習與從頭訓練 CNN 的表現有何差異？

## 執行環境

本專案使用上層資料夾既有的 **WSL Python 3.10 venv**。請在 WSL 終端機執行：

```bash
cd /mnt/c/Users/orange/Desktop/class/machine_learning
source .venv/bin/activate
python midterm/project.py prepare
python midterm/project.py train --model cnn --epochs 15
python midterm/project.py train --model resnet18 --epochs 15
python midterm/train_dnntc.py
python midterm/project.py evaluate
python midterm/make_assets.py
python midterm/write_report.py
python midterm/demo.py
```

瀏覽器開啟 http://localhost:8765。Demo 僅在本機運作，需先完成訓練。
Windows 也可以直接雙擊本目錄的 `run_demo.cmd`，等待模型載入後再開啟網址。
首次執行需要網路下載資料集、ResNet18 與 ResNeXt-101 預訓練權重，完成後 Demo 可離線執行。
預設 CPU 訓練，避免舊版 PyTorch 與新 GPU 架構不相容。只有通過實際 CUDA 運算測試後才加上 `--device cuda`。

全新環境可使用 Python 3.10 建立 venv，再安裝本目錄 requirements.txt。
原專案根目錄的 requirements.txt 是另一組環境，不要混用。

## 方法與公平性

- 資料來源：[TrashNet 作者儲存庫](https://github.com/garythung/trashnet) 與 [作者資料集](https://huggingface.co/datasets/garythung/trashnet)。請保留作者 Gary Thung 與 Mindy Yang 的引用。資料頁標示 MIT 授權。
- 六類：紙板、玻璃、金屬、紙張、塑膠、其他垃圾。這些是影像標籤，不等同臺灣各地回收規則。
- 先以解碼後影像 SHA-256 檢查完全重複圖片；本次發現三組同圖異標籤，共排除六張圖片。其餘 2,521 張固定 seed=42 做約 70/15/15 分層切分。
- 完全重複檢查不能排除同一物件的不同角度或近似圖片；結果只代表此影像切分，不保證跨物件泛化。
- 三模型共用切分、160×160 輸入、分類頭 batch size=32、AdamW、15 epochs。DNN-TC 使用兩組固定增強特徵，其餘模型每輪重新取樣增強。
- CNN 從頭訓練。ResNet18 使用 ImageNet 權重、凍結骨幹與 BatchNorm 統計，只訓練最後分類層。
- DNN-TC 比較版使用 ImageNet V1 ResNeXt-101 32x8d，凍結骨幹，訓練 2048→1024→6 的分類頭。
- 此實驗比較三種實作策略，同時改變架構、預訓練和增強流程，不能將差異全部歸因於單一因素。若要隔離因素，需要另設同架構對照。
- 以 validation macro F1 選擇 checkpoint。測試集僅於訓練完成後評估，不用於調參。
- Accuracy 容易偏重多數類，因此同時報告 macro F1、各類 recall 與混淆矩陣。
- 單次 seed 實驗，沒有跨種子信賴區間。白背景資料與生活環境存在分布差異。

## 輸出

`data/splits.json` 記錄每張圖片的分組與雜湊。
`outputs/dataset.json` 記錄來源、下載檔雜湊、去重與類別統計。
每個模型的資料夾包含 `best.pt`、`history.json`、`training.json`、`test.json`、`predictions.csv`、`learning.png` 與 `confusion.png`。
`loss.png` 是獨立的 train／validation loss 圖，虛線標記選定的 epoch。
`outputs/summary.json` 是最終測試結果。沒有此檔時，不應宣稱實驗已完成。

三模型成績以 `實驗結果.md` 和 `outputs/summary.json` 為準。DNN-TC 為參考論文架構的凍結骨幹比較版，並非完整論文復現，詳細設定見 `DNN_TC_METHOD.md`。
各類表現可能不同，不能只憑整體 Accuracy 宣稱某模型在所有類別都較好。
`程式導讀.md` 解釋核心程式；`QA與展示流程.md` 提供問答與報告準備流程。
`check_project.py` 檢查資料互斥與模型儲存；`check_demo.py` 測試真實圖片的 HTTP 推論流程。

## 15 分鐘報告與 5 分鐘 QA

動機 1 分鐘、資料 2 分鐘、方法 3 分鐘、結果 4 分鐘、Demo 3 分鐘、結論 2 分鐘。
Demo 展示測試集中的成功與失敗案例，說明它們是已保留的測試圖片。另拍攝生活圖片時，明確區分它們是額外展示，沒有計入正式測試準確率。

## 重要程式

`project.py`：資料下載與切分、模型、訓練、評估。
`train_dnntc.py`：DNN-TC 凍結骨幹、兩組固定增強特徵與分類頭訓練。
`demo.py`：本機圖片上傳與三模型推論。
模型檔只能載入本專案自行產生的可信 checkpoint。
