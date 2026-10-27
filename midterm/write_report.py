"""Write a report using only actual saved metrics."""
import json
from project import ROOT, OUT, CLASSES, MODEL_NAMES

summary = json.loads((OUT / 'summary.json').read_text())
lines = ['# 實驗結果', '', '本表來自固定保留的 379 張測試圖片，沒有以測試成績挑選 checkpoint。', '',
         '| 模型 | Accuracy | Macro F1 | 選定 epoch | 訓練秒數 | 可訓練參數 |',
         '|---|---:|---:|---:|---:|---:|']
for name, scores in summary.items():
    test = json.loads((OUT / name / 'test.json').read_text())
    meta = json.loads((OUT / name / 'training.json').read_text())
    lines.append(f'| {name} | {scores["accuracy"]:.2%} | {scores["macro_f1"]:.4f} | '
                 f'{test["selected_epoch"]} | {meta["seconds"]:.1f} | {meta["trainable_parameters"]:,} |')
lines += ['', '訓練秒數包含 epoch 中的資料讀取與驗證，不含首次下載權重、模型初始化與最後測試評估。',
          '這是本機 CPU 單次量測，受其他程序與硬體影響。', '', '## 各類別 Recall', '',
          '| 類別 | CNN | ResNet18 | DNN-TC 比較版 | 測試張數 |', '|---|---:|---:|---:|---:|']
reports = {name: json.loads((OUT / name / 'test.json').read_text())['report'] for name in summary}
for label in CLASSES:
    lines.append(f'| {label} | {reports["cnn"][label]["recall"]:.2%} | '
                 f'{reports["resnet18"][label]["recall"]:.2%} | {reports["dnn_tc"][label]["recall"]:.2%} | '
                 f'{int(reports["cnn"][label]["support"])} |')
delta = (summary['resnet18']['accuracy'] - summary['cnn']['accuracy']) * 100
lines += ['', '## 解讀', '', f'ResNet18 的測試 Accuracy 比 CNN 高 {delta:.2f} 個百分點。',
          f'DNN-TC 架構比較版 Accuracy 為 {summary["dnn_tc"]["accuracy"]:.2%}，Macro F1 為 {summary["dnn_tc"]["macro_f1"]:.4f}。',
          '本比較同時改變模型架構、預訓練與訓練流程，不能把差距全部歸因於任何單一因素。',
          '只有一次固定 seed 實驗，未測量跨次訓練波動，亦未證明真實回收場域的泛化能力。', '',
          '## DNN-TC 架構比較版與論文的差異', '',
          '採用 TorchVision ResNeXt-101 32x8d ImageNet V1 骨幹，分類頭 2048 → 1024 → 6，中間使用 ReLU。',
          '本實作凍結骨幹，預先擷取兩組固定增強圖片的特徵，交替訓練分類頭 15 輪；測試集未參與特徵訓練或模型選擇。',
          '論文使用 224px、60/20/20 切分及 Adam/SGD 微調流程；本專案使用 160px、既有 70/15/15 切分和 AdamW。',
          '因此這是參考架構的比較實作，不是完整論文復現。時間包含特徵擷取與分類頭訓練，快取讓訓練時間不能直接視為骨幹計算速度。',
          '論文來源：https://doi.org/10.1109/ACCESS.2019.2959033', '',
          '## Loss 曲線', '', '![CNN Loss](outputs/cnn/loss.png)', '',
          '![ResNet18 Loss](outputs/resnet18/loss.png)', '',
          '![DNN-TC Loss](outputs/dnn_tc/loss.png)', '',
          'Loss 使用 cross entropy，虛線標示以驗證集 macro F1 選定的 epoch。',
          '訓練 loss 包含資料增強與模型更新過程，驗證 loss 則以該輪結束後模型計算。', '',
          '## 其他圖表', '', '![CNN 學習曲線](outputs/cnn/learning.png)', '',
          '![ResNet18 學習曲線](outputs/resnet18/learning.png)', '',
          '![CNN 混淆矩陣](outputs/cnn/confusion.png)', '',
          '![ResNet18 混淆矩陣](outputs/resnet18/confusion.png)', '',
          '![DNN-TC 混淆矩陣](outputs/dnn_tc/confusion.png)', '',
          '資料來源：https://github.com/garythung/trashnet']
(ROOT / 'comparison.json').write_text(json.dumps({n: {
    'scores': summary[n], 'training': json.loads((OUT / n / 'training.json').read_text()),
    'history': json.loads((OUT / n / 'history.json').read_text()),
    'report': reports[n], 'selected_epoch': json.loads((OUT / n / 'test.json').read_text())['selected_epoch']
} for n in MODEL_NAMES}, ensure_ascii=False), encoding='utf-8')
(ROOT / '實驗結果.md').write_text('\n'.join(lines), encoding='utf-8')
print('Wrote 實驗結果.md')
