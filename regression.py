import torch
from torch import nn, optim
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

CSV_TAXI_FARE = 'taxi_fare_training.csv'
torch.manual_seed(0)
data = pd.read_csv(CSV_TAXI_FARE)
# n = 100
x = torch.tensor(data['distance_km'].to_numpy(),dtype=torch.float32).reshape(-1,1)
y = torch.tensor(data['fare_ntd'].to_numpy(),dtype=torch.float32).reshape(-1,1)

net = Model()

opt = optim.SGD(net.parameters(), lr=0.01)
loss_f = nn.MSELoss()
loss_hist = []

OUT_DIR = "result"
os.makedirs(OUT_DIR, exist_ok = True)



for epoch in range(50):
    y_hat = net(x)
    loss = loss_f(y, y_hat)
    loss.backward()
    opt.step()
    opt.zero_grad()
    loss_hist.append(float(loss.item()))
    print(f'epoch:{epoch}, loss:{loss.item()}')

plt.figure()
plt.plot(range(len(loss_hist)), loss_hist, marker='o')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
loss_curve_path = os.path.join(OUT_DIR, 'loss.png')
plt.savefig(loss_curve_path, dpi=150)
plt.close()

net.eval()
with torch.no_grad():
    y_hat = net(x)

arr = np.hstack(
    [
        x.detach().cpu().numpy(),
        y.detach().cpu().numpy(),
        y_hat.detach().cpu().numpy()
    ])
pred_csv_path = os.path.join(OUT_DIR, 'predictions.csv')
np.savetxt(pred_csv_path, arr, delimiter=',', header='x,y,y_hat', comments='')

idx = x.squeeze(1).argsort()
x_sorted = x[idx]
yhat_sorted = y_hat[idx]

plt.figure()
plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), 'o', alpha=0.6, label='Data (x, y)')
plt.plot(x.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), 'x', alpha=0.8, label='Model output on training x')
plt.plot(x_sorted.detach().cpu().numpy(), yhat_sorted.detach().cpu().numpy(), '-', alpha=0.6, linewidth=2, label='Connected model outputs')

plt.xlabel('distance_km')
plt.ylabel('fare_ntd')
plt.title('Taxi Fare: Data vs Model Outputs')
plt.legend()
plt.tight_layout()
scatter_outputs_path = os.path.join(OUT_DIR,'data_with_true_model_outputs.png')
plt.savefig(scatter_outputs_path, dpi=150)
plt.close()