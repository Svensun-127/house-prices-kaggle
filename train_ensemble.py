import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

SEED = 42

# Load data
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
y_train = np.load('y_train.npy', allow_pickle=True)
y_val = np.load('y_val.npy', allow_pickle=True)

feature_names = X_train.columns.tolist()


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ---- Model 1: XGBoost (with regularization) ----
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.03,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'seed': SEED,
}
xgb_bst = xgb.train(
    xgb_params, dtrain, num_boost_round=1000,
    evals=[(dval, 'validation')],
    early_stopping_rounds=50, verbose_eval=False,
)
xgb_val_pred = xgb_bst.predict(dval)

# ---- Model 2: LightGBM (with regularization) ----
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.03,
    'max_depth': 4,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'seed': SEED,
    'verbose': -1,
}
lgb_bst = lgb.train(
    lgb_params, lgb_train, num_boost_round=1000,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)
lgb_val_pred = lgb_bst.predict(X_val)

# ---- Model 3: CatBoost ----
cb_bst = CatBoostRegressor(
    learning_rate=0.03, depth=4, iterations=1000,
    subsample=0.8, random_seed=SEED, verbose=False,
)
cb_bst.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
cb_val_pred = cb_bst.predict(X_val)

# ---- Individual RMSLE ----
xgb_rmsle = rmsle(y_val, xgb_val_pred)
lgb_rmsle = rmsle(y_val, lgb_val_pred)
cb_rmsle = rmsle(y_val, cb_val_pred)

print('Individual validation RMSLE:')
print(f'  XGBoost:  {xgb_rmsle:.6f}')
print(f'  LightGBM: {lgb_rmsle:.6f}')
print(f'  CatBoost: {cb_rmsle:.6f}')

# ---- Ensemble: Simple Average ----
avg_pred = (xgb_val_pred + lgb_val_pred + cb_val_pred) / 3
avg_rmsle = rmsle(y_val, avg_pred)
print(f'\nSimple average RMSLE: {avg_rmsle:.6f}')

# ---- Ensemble: Optimized weighted average (min 0.1 per model) ----
val_preds = np.column_stack([xgb_val_pred, lgb_val_pred, cb_val_pred])


def ensemble_rmsle(weights):
    w = np.array(weights)
    w = np.maximum(w, 1e-8) / np.maximum(w, 1e-8).sum()
    blended = val_preds @ w
    return rmsle(y_val, blended)


init_weights = [1/3, 1/3, 1/3]
bounds = [(0.1, 0.8), (0.1, 0.8), (0.1, 0.8)]
constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})

result = minimize(ensemble_rmsle, init_weights, bounds=bounds, constraints=constraints, method='SLSQP')
opt_weights = np.maximum(result.x, 1e-8)
opt_weights = opt_weights / opt_weights.sum()
opt_rmsle = ensemble_rmsle(opt_weights)

print(f'\nOptimized ensemble weights (min 0.1 each):')
print(f'  XGBoost:  {opt_weights[0]:.4f}')
print(f'  LightGBM: {opt_weights[1]:.4f}')
print(f'  CatBoost: {opt_weights[2]:.4f}')
print(f'\nWeighted ensemble RMSLE: {opt_rmsle:.6f}')
print(f'Best single RMSLE: {min(xgb_rmsle, lgb_rmsle, cb_rmsle):.6f}')
print(f'Ensemble vs best single improvement: {min(xgb_rmsle, lgb_rmsle, cb_rmsle) - opt_rmsle:.6f}')

# Use the better of simple average or optimized
best_weights = opt_weights if opt_rmsle < avg_rmsle else np.array([1/3, 1/3, 1/3])
best_rmsle = min(opt_rmsle, avg_rmsle)
print(f'\nFinal ensemble RMSLE (best of avg/weighted): {best_rmsle:.6f}')

# ---- Save models and weights ----
np.save('ensemble_weights.npy', best_weights)
xgb_bst.save_model('xgb_model.json')
lgb_bst.save_model('lgb_model.txt')
cb_bst.save_model('cb_model.cbm')

print('Models and weights saved.')
