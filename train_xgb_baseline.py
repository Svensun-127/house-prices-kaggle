import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load data
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
y_train = np.load('y_train.npy', allow_pickle=True)
y_val = np.load('y_val.npy', allow_pickle=True)

# Train XGBoost model using native API for early stopping
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 5,
    'seed': 42,
}

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_train.columns.tolist())

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dval, 'validation')],
    early_stopping_rounds=50,
    verbose_eval=False,
)

# Predictions
train_pred = bst.predict(dtrain)
val_pred = bst.predict(dval)

rmsle_train = np.sqrt(mean_squared_error(y_train, train_pred))
rmsle_val = np.sqrt(mean_squared_error(y_val, val_pred))

feat_imp = pd.Series(bst.get_score(importance_type='gain')).sort_values(ascending=False)

print(f'RMSLE train: {rmsle_train:.6f}')
print(f'RMSLE val: {rmsle_val:.6f}')
print('\nTop 10 feature importances:')
for name, score in feat_imp.head(10).items():
    print(f'{name}: {score:.6f}')
