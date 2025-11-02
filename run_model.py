import os
import glob
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# 랜덤 시드 고정
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# LSTM 모델 정의
class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        input_size: 입력 feature 수
        hidden_size: LSTM hidden 차원
        num_layers: LSTM 층 수
        output_size: 예측값 차원
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_size]
        """
        out, _ = self.lstm(x)       # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]          # 마지막 시점 출력
        out = self.fc(out)           # [batch, output_size]
        return out
    

# 시계열 데이터셋 클래스
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# 데이터 불러오기
file_path = "data"
file_name = "fst_포인트5_통합_6_8_10_기상병합"
df = pd.read_csv(f"{file_path}/{file_name}.csv")

# 날짜 제거 (index로 쓰거나 feature에서 제외)
df['날짜'] = pd.to_datetime(df['날짜'])
df = df.sort_values('날짜')
try:
    features = df.drop(columns=['날짜', '날짜 시간', 'TM']).values
except:
    features = df.drop(columns=['날짜', 'TM']).values
print("Features shape:", features.shape)
print("First 3 rows:\n", features[:3])

# 정규화
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Chlorophyll-a 농도 컬럼 인덱스 (features 기준)
chl_flu_idx = 16
chl_nong_idx = 17

# 시퀀스 생성
seq_len = 5
X, y = [], []
for i in range(len(scaled_features) - seq_len):
    X.append(scaled_features[i:i+seq_len])
    next_day = scaled_features[i + seq_len]
    y.append(next_day[[chl_flu_idx, chl_nong_idx]])  # 두 값 예측

X, y = np.array(X), np.array(y)
print("X shape:", X.shape, "y shape:", y.shape)

# 데이터셋 분할
train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.1)
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
print(X_train[0], y_train[0])

# 데이터로더 생성
train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=8, shuffle=True)
val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=8, shuffle=False)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=8, shuffle=False)
print(f"train_loader batches: {len(train_loader)}, val_loader batches: {len(val_loader)}, test_loader batches: {len(test_loader)}")

# 모델 정의
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 3  # 1: Vanilla LSTM, 2>: Stacked LSTM
output_size = 2
model = VanillaLSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 300
best_train_loss = float('inf')
best_val_loss = float('inf')
best_epoch = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 검증
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            preds = model(X_val)
            val_loss += criterion(preds, y_val).item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {total_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")\
    
    if val_loss < best_val_loss:
        best_train_loss = total_loss/len(train_loader)
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        best_epoch = epoch + 1

model.load_state_dict(best_model_state)
print(f"✅ {best_epoch} model loaded with Val Loss: {best_val_loss:.6f}")

time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
weight_dir = "weights"
weight_name = f"{time_str}_{file_name}_{best_epoch}epoch.pth"
torch.save(model.state_dict(), f"{weight_dir}/{weight_name}")
print(f"Saved: {weight_dir}/{weight_name}")

# 테스트 및 평가
model.eval()
all_preds = []
all_labels = []
test_loss = 0

with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test = X_test.float()
        y_test = y_test.float()

        y_preds = model(X_test)
        all_preds.append(y_preds.cpu())
        all_labels.append(y_test.cpu())
        test_loss += criterion(y_preds, y_test).item()

y_pred = torch.cat(all_preds, dim=0).numpy()
y_true = torch.cat(all_labels, dim=0).numpy()

mse_each = ((y_pred - y_true) ** 2).mean(axis=0)
print(f"MSE per feature → Chl_flu: {mse_each[0]:.6f}, Chl_nong: {mse_each[1]:.6f}")

print("\n=== 마지막 입력 및 예측 결과 ===")
print(f"입력 시퀀스 (마지막 샘플):\n{X_test[-1].numpy()}")
print(f"실제값 (y_true): {y_true[-1]}")
print(f"예측값 (y_pred): {y_pred[-1]}")

# Visualization
feature_names = ["Chl_flu", "Chl_nong"]
plt.figure(figsize=(12, 5))

for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.plot(y_true[:, i], label="Actual", marker='o', alpha=0.8)
    plt.plot(y_pred[:, i], label="Predicted", marker='x', alpha=0.8)
    plt.title(f"{feature_names[i]} Prediction")
    plt.xlabel("Sample (Time order)")
    plt.ylabel(feature_names[i])
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# Scatter comparison
plt.figure(figsize=(6,6))
for i in range(2):
    plt.scatter(y_true[:, i], y_pred[:, i], label=feature_names[i], alpha=0.7)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="y = x")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.legend()
plt.title("Prediction vs Actual Scatter (Both Outputs)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 마지막 샘플 실제값과 예측값 복원
feature_indices = [chl_flu_idx, chl_nong_idx]

dummy_true = np.zeros(scaled_features.shape[1])
dummy_pred = np.zeros(scaled_features.shape[1])
dummy_true[feature_indices] = y_true[-1]  # 스케일된 마지막 실제값
dummy_pred[feature_indices] = y_pred[-1]  # 스케일된 마지막 예측값

# 역변환
y_true_orig = scaler.inverse_transform(dummy_true.reshape(1, -1))[0, feature_indices]
y_pred_orig = scaler.inverse_transform(dummy_pred.reshape(1, -1))[0, feature_indices]
print("\n=== 마지막 입력 및 예측 결과 (원래 단위) ===")
print(f"실제값 (원래 단위): {y_true_orig}")
print(f"예측값 (원래 단위): {y_pred_orig}")

# 결과 로그 저장
log_dict = {
    "datetime": time_str,
    "file_name": file_name,
    "best_epoch": best_epoch,
    "train_loss": best_train_loss,
    "val_loss": best_val_loss,
    "test_loss": test_loss,
    "test_mse": np.mean(mse_each),
    "hidden_size": hidden_size,
    "num_layers": num_layers
}
with open('weights/weights_result.csv', mode='a', newline='', encoding='utf-8') as f:
    df_new = pd.DataFrame([log_dict])
    df_new.to_csv('weights/weights_result.csv', mode='a', index=False, header=False, encoding='utf-8')
    