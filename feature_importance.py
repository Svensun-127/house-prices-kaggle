"""Analyze feature importance from a trained XGBoost model."""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import inv_boxcox
from sklearn.metrics import mean_squared_error
import os

SEED = 42

X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
bc_lambda = float(np.load('boxcox_lambda.npy'))
feature_names = X_train.columns.tolist()

# Train a strong XGBoost for feature importance
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.02,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'seed': SEED,
}

bst = xgb.train(params, dtrain, num_boost_round=2000,
                evals=[(dval, 'validation')],
                early_stopping_rounds=100, verbose_eval=50)

# Feature importance (gain)
importance = bst.get_score(importance_type='gain')
importance_series = pd.Series(importance).sort_values(ascending=False)

print('=== Top 15 Feature Importances (by Gain) ===')
for i, (name, score) in enumerate(importance_series.head(15).items(), 1):
    print(f'  {i:2d}. {name:<30s} {score:10.2f}')

# Validation RMSLE
val_pred_boxcox = bst.predict(dval)
y_val_raw = inv_boxcox(y_val, bc_lambda)
val_pred_raw = inv_boxcox(val_pred_boxcox, bc_lambda)
val_pred_raw = np.maximum(val_pred_raw, 0)
rmsle_val = np.sqrt(mean_squared_error(np.log1p(y_val_raw), np.log1p(val_pred_raw)))
print(f'\nValidation RMSLE (original space): {rmsle_val:.6f}')

# Check if validation RMSLE is below target
if rmsle_val < 0.126:
    print('Target achieved: RMSLE < 0.126')
else:
    print(f'Gap to target: {rmsle_val - 0.126:.6f}')
