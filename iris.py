import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
OUTPUT_DIR = "iris_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

iris = load_iris()
print("Original: ", iris.data[0], iris.target[0])

X = iris.data.astype(np.float32)
y = iris.target.astype(np.int64)
print(f"Data Shape X: {X.shape}, y: {y.shape}")
print(f"Data: X: {X[0]}, y:{y[0]}")

feature_names = iris.feature_names
class_names = iris.target_names.tolist()
print(f"Names: \n{iris.feature_names}, \n {iris.target_names}, \n {iris.target_names.tolist()}\n")
print(f"Names: \n{type(iris.feature_names)}, \n {type(iris.target_names)}, \n {type(iris.target_names.tolist())}\n")

df = pd.DataFrame(X, columns=feature_names)
df["target"] = [class_names[label] for label in y]
print("=== Iris Dataset ===")
print(df.head())
print(f"\n Data Counts:{len(df)}, Feature Counts:{X.shape[1]}，classes:{', '.join(class_names)}")

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)
print(f"Data Split:train={len(y_train)}, val={len(y_val)},test={len(y_test)}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

BATCH_SIZE = 16

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class DNN(nn.Module):
    def __init__(self, in_dim=4, hidden1=64, hideen2=32, out_dim=3, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hideen2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hideen2, out_dim),
        )
    def forward(self,x):
        return self.net(x)

LEARNING_RATE = 1e-2
MAX_EPOCHS = 200
PATIENCE = 15
model = DNN().to(device)

import torchinfo

print(X_train.shape,'\n')
torchinfo.summary(model, input_size=(BATCH_SIZE, X_train.shape[1]), device=device.type)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = 0.9)
print(f"\n使用裝置:{device}")

if os.path.exists(OUTPUT_DIR + "/best.pt"):
    print("best.pt is exist, so skip training")
else:
    print("best.pt is not exist, start training")
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_loss, best_epoch, bad_epochs = float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
           X_batch, y_batch = X_batch.to(device), y_batch.to(device)
           optimizer.zero_grad()
           logits = model(X_batch)
           loss = criterion(logits, y_batch)
           loss.backward()
           optimizer.step()
           train_loss_sum += loss.item() * len(y_batch)
           train_correct += (logits.argmax(dim=1) == y_batch).sum().item();train_total += len(y_batch)
        train_loss = train_loss_sum / train_total; train_accuracy = train_correct / train_total

        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_loss_sum += criterion(logits, y_batch).item() * len(y_batch)
                val_correct += (logits.argmax(dim=1) == y_batch).sum().item(); val_total += len(y_batch)
        val_loss = val_loss_sum / val_total; val_accuracy = val_correct / val_total
        train_losses.append(train_loss); val_losses.append(val_loss); train_accuracies.append(train_accuracy); val_accuracies.append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save(model.state_dict(), OUTPUT_DIR + "/best.pt")
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train loss={train_loss:.4f} | train acc={train_accuracy:.3f} | val loss={val_loss:.4f} | val acc={val_accuracy:.3f}")
        if bad_epochs >= PATIENCE:
            print(f"Early stop:epoch{epoch}") 
            break
        
    epochs = range(1, len(train_losses) +1);fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_losses, label="Train loss")
    axes[0].plot(epochs, val_losses, label="Validation loss")
    axes[0].set(title="Loss Curve", xlabel="Epoch", ylabel="Cross-entropy loss");axes[0].legend()
    axes[1].plot(epochs, train_accuracies, label="Train accuracy")
    axes[1].plot(epochs, val_accuracies, label="Validation accuracy")
    axes[1].set(title="Accuracy Curve", xlabel="Epoch", ylabel="Accuracy");axes[1].legend();fig.tight_layout()
    loss_path = OUTPUT_DIR + "/loss_curve.png"; fig.savefig(loss_path, dpi=150); plt.close(fig)

model.load_state_dict(torch.load(OUTPUT_DIR + "/best.pt", map_location=device))
model.eval();y_true, y_pred=[], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        logits = model(X_batch.to(device))
        y_true.extend(y_batch.numpy())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy())
print("\n=== Test Evaluation ===")
print(f"Test Accuracy:{accuracy_score(y_true, y_pred):.3f}")
print(classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(cmap="Blues", values_format="d")
plt.title("Iris Confusion Matrix"); plt.tight_layout()
cm_path = OUTPUT_DIR + "/confusion_matrix.png"
plt.savefig(cm_path, dpi=150);plt.close()

predictions = pd.DataFrame(X_test, columns=feature_names)
predictions["true_label"] = [class_names[label] for label in y_true]
predictions["predicted_label"] = [class_names[label] for label in y_pred]
predictions_path = OUTPUT_DIR + "/predictions.csv"
predictions.to_csv(predictions_path, index=False, float_format="%.4f")
