import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Učitavanje podataka i enkodovanje labela (Cartman - 1, Stan - 2 ...)
df = pd.read_csv('./data/south_park_train.csv')
df = df.dropna()
df['Character'] = LabelEncoder().fit_transform(df['Character'])

df.head()

# ---------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(df['Line'], df['Character'], test_size=0.2, random_state=42)
X_train.head()

# ---------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer(stop_words="english")

# Ponovo obratiti pažnju na razliku fit_transform i transform
X_train = bow.fit_transform(X_train)
X_test = bow.transform(X_test)

# pogledajmo kako izgleda naučeni rečnik
bow.vocabulary_

# ---------------------------------------------

# Biramo model, npr. Naivni Bayes
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB().fit(X_train, y_train)
y_pred = nb.predict(X_test)

# evaluiramo model
from sklearn.metrics import accuracy_score
print(f'Accuracy: {accuracy_score(y_pred, y_test)}')

# ---------------------------------------------
# ---------------------------------------------

# Ili npr. ANN klasifikator
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLPClassifier, self).__init__()
        self.relu = nn.ReLU()
        sizes = [input_size] + hidden_sizes + [num_classes]
        self.layers = nn.ModuleList([nn.Linear(sizes[i - 1], sizes[i]) for i in range(1, len(sizes))])

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.relu(out)
        return out

X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)

input_size = X_train.shape[1]
hidden_sizes = [512, 256, 128] # Ovde možete eksperimentisati sa različitim brojevima neurona po sloju
num_classes = len(set(y_train)) # Broj jedinstvenih klasa
model = MLPClassifier(input_size, hidden_sizes, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treniranje modela
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Evaluacija
with torch.no_grad():
    model.eval()
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(y_pred=predicted, y_true=y_test)
    print(f'Accuracy: {accuracy}')