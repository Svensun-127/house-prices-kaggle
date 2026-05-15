import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
y_train = np.load('y_train.npy', allow_pickle=True)
y_val = np.load('y_val.npy', allow_pickle=True)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

# RMSLE on log-transformed target = RMSE on log target
rmsle_train = np.sqrt(mean_squared_error(y_train, train_pred))
rmsle_val = np.sqrt(mean_squared_error(y_val, val_pred))

# Feature importances
feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
feat_imp = feat_imp.sort_values(ascending=False)

print(f'RMSLE train: {rmsle_train:.6f}')
print(f'RMSLE val: {rmsle_val:.6f}')
print('\nTop 10 feature importances:')
for name, score in feat_imp.head(10).items():
    print(f'{name}: {score:.6f}')
