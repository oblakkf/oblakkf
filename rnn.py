import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np

# -----------------------------
# Konfiguracija
# -----------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PROBLEM_TYPE = 'regression'  # ili 'classification'
INPUT_SIZE = 1               # broj input feature-a po vremenskom koraku
HIDDEN_SIZE = 128
OUTPUT_SIZE = 1 if PROBLEM_TYPE == 'regression' else 10  # npr. 10 klasa
SEQUENCE_LENGTH = 20
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001
EVAL_EVERY = 10

# -----------------------------
# Model
# -----------------------------

class GeneralRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, problem_type='regression'):
        super(GeneralRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.problem_type = problem_type

    def forward(self, x):
        out, hidden = self.rnn(x)  # out: [B, T, H]
        
        if self.problem_type == 'regression':
            out = self.fc(out)     # [B, T, 1] - sekvencijalna regresija
        else:
            out = self.fc(out[:, -1, :])  # klasifikacija cele sekvence
        return out

# -----------------------------
# Dummy podaci (zameniti pravim)
# -----------------------------

# Primer dummy podataka: 1000 sekvenci dužine 20
X = torch.randn(1000, SEQUENCE_LENGTH, INPUT_SIZE)
if PROBLEM_TYPE == 'regression':
    y = torch.randn(1000, SEQUENCE_LENGTH, OUTPUT_SIZE)
else:
    y = torch.randint(0, OUTPUT_SIZE, (1000,))  # klasifikacija cele sekvence

# Dataset i DataLoader
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Inicijalizacija modela i loss-a
# -----------------------------

model = GeneralRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, PROBLEM_TYPE).to(DEVICE)

if PROBLEM_TYPE == 'regression':
    loss_fn = nn.MSELoss()
else:
    loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Trening petlja
# -----------------------------

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        output = model(X_batch)

        if PROBLEM_TYPE == 'regression':
            loss = loss_fn(output, y_batch)
        else:
            loss = loss_fn(output, y_batch)  # y_batch: (B,), output: (B, num_classes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    if epoch % EVAL_EVERY == 0 or epoch == 1:
        avg_loss = train_loss / len(loader)
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

# -----------------------------
# Evaluacija modela
# -----------------------------

model.eval()
with torch.no_grad():
    X_test = X[:100].to(DEVICE)

    if PROBLEM_TYPE == 'regression':
        y_test = y[:100].to(DEVICE)
        y_pred = model(X_test)
        rmse = torch.sqrt(loss_fn(y_pred, y_test))
        print(f"Test RMSE: {rmse:.4f}")
    else:
        y_test = y[:100].to(DEVICE)
        y_pred = model(X_test)
        acc = (y_pred.argmax(dim=1) == y_test).float().mean()
        print(f"Test Accuracy: {acc:.4f}")
