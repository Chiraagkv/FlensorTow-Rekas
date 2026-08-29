import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from rekas.Sequential import Sequential
from rekas.layers import Input, Dense
from rekas.activations import ReLU, Softmax
from rekas.losses import CCE
from rekas.optimizers import Adam

print("LOADING DATA")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data.astype(np.float32), mnist.target.astype(int)

# Subset
indices = np.random.choice(len(X), 5000, replace=False)
X = X.iloc[indices].values
y = y.values[indices]


X /= 255.0

y_onehot = np.zeros((len(y), 10))
y_onehot[np.arange(len(y)), y] = 1


X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
y_test_labels = np.argmax(y_test, axis=1)

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"Input Features: {X_train.shape[1]} (28x28 pixels)\n")

model = Sequential([
    Input(784),
    Dense(64),
    ReLU(),
    Dense(10),
    Softmax()
])

model.compile(CCE(), Adam(0.001, 0.9, 0.999, 16))

print("TRAINING MODEL")
model.fit(X_train, y_train, epochs=50, verbose=False)


probs = model.predict(X_test)
preds = np.argmax(probs, axis=1)

acc = accuracy_score(y_test_labels, preds)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
