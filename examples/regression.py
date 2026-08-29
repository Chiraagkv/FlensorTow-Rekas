from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from rekas.layers import Input, Dense
from rekas.losses import MSE
from rekas.activations import ReLU
from rekas.optimizers import Nesterov
from rekas.Sequential import Sequential

data = load_diabetes()
X, y = data.data, data.target

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"Input Features: {X_train.shape[1]}\n")

model = Sequential([
    Input(10),
    Dense(16),
    ReLU(),
    Dense(8),
    ReLU(),
    Dense(1)
])

model.compile(MSE(), Nesterov(0.05, 0.1, batch_size=8))

model.fit(X_train, y_train, epochs=50, verbose=True)

preds_scaled = model.predict(X_test)

preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mse = mean_squared_error(y_test_orig, preds)
r2 = r2_score(y_test_orig, preds)

print(f"Test MSE: {mse:.2f}")
print(f"Test R² Score: {r2:.4f}")