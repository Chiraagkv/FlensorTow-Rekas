import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from rekas.Sequential import Sequential
from rekas.layers import Input, Dense
from rekas.activations import ReLU, Sigmoid
from rekas.losses import BCE
from rekas.optimizers import RMSProp

print("LOAD DATA")
data = load_breast_cancer()
X, y = data.data, data.target 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 ) 

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"Input Features: {X_train.shape[1]}\n") 

scaler_x = StandardScaler()

X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"Input Features: {X_train.shape[1]}\n")

model = Sequential([
    Input(30),
    Dense(16),
    ReLU(),
    Dense(8),
    ReLU(),
    Dense(1),
    Sigmoid()
])

model.compile(BCE(), RMSProp(0.05, 0.2, 16))

print("TRAINING MODEL")
model.fit(X_train, y_train, epochs=30, verbose=True)

print("\n\n")
probs = model.predict(X_test)

preds = (probs >= 0.5).astype(int)

acc = accuracy_score(y_test, preds)
print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, preds, target_names=data.target_names))