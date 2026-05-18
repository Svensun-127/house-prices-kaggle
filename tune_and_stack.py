"""Quick Optuna tuning + Stacking: RMSLE computed in original SalePrice space."""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.special import inv_boxcox
import optuna
import json
import os
import warnings
warnings.filterwarnings('ignore')

SEED = 42
N_TRIALS = 50

# Load data
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
y_train_boxcox = np.load('y_train.npy')
y_val_boxcox = np.load('y_val.npy')
bc_lambda = float(np.load('boxcox_lambda.npy'))
feature_names = X_train.columns.tolist()

# Inverse transform to original SalePrice for proper RMSLE
y_train_raw = inv_boxcox(y_train_boxcox, bc_lambda)
y_val_raw = inv_boxcox(y_val_boxcox, bc_lambda)


def rmsle_raw(y_true_raw, y_pred_boxcox):
    """Compute RMSLE in original SalePrice space."""
    y_pred_raw = inv_boxcox(y_pred_boxcox, bc_lambda)
    y_pred_raw = np.maximum(y_pred_raw, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true_raw), np.log1p(y_pred_raw)))


# ============================================================
# Quick Optuna Tuning for XGBoost (30 trials)
# ============================================================
print('=== Tuning XGBoost (30 trials) ===')


def xgb_objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': trial.suggest_float('learning_rate', 0.008, 0.04, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.65, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'seed': SEED,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train_boxcox, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val_boxcox, feature_names=feature_names)
    bst = xgb.train(params, dtrain, num_boost_round=2000,
                    evals=[(dval, 'validation')],
                    early_stopping_rounds=100, verbose_eval=False)
    pred = bst.predict(dval)
    return rmsle_raw(y_val_raw, pred)


xgb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
xgb_study.optimize(xgb_objective, n_trials=30, show_progress_bar=True)
print(f'Best XGBoost RMSLE (orig space): {xgb_study.best_value:.6f}')
print(f'Best params: {xgb_study.best_params}')

# ============================================================
# Quick Optuna Tuning for LightGBM (30 trials)
# ============================================================
print('\n=== Tuning LightGBM (30 trials) ===')


def lgb_objective(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.008, 0.04, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 8, 63),
        'subsample': trial.suggest_float('subsample', 0.65, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 40),
        'seed': SEED, 'verbose': -1,
    }
    lgb_tr = lgb.Dataset(X_train, label=y_train_boxcox)
    lgb_va = lgb.Dataset(X_val, label=y_val_boxcox, reference=lgb_tr)
    bst = lgb.train(params, lgb_tr, num_boost_round=2000,
                    valid_sets=[lgb_va],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    pred = bst.predict(X_val)
    return rmsle_raw(y_val_raw, pred)


lgb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
lgb_study.optimize(lgb_objective, n_trials=30, show_progress_bar=True)
print(f'Best LightGBM RMSLE (orig space): {lgb_study.best_value:.6f}')
print(f'Best params: {lgb_study.best_params}')

# ============================================================
# Quick Optuna Tuning for CatBoost (30 trials)
# ============================================================
print('\n=== Tuning CatBoost (30 trials) ===')


def cb_objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.008, 0.04, log=True),
        'depth': trial.suggest_int('depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.65, 0.95),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 5.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
    }
    model = CatBoostRegressor(**params, iterations=2000, random_seed=SEED, verbose=False)
    model.fit(X_train, y_train_boxcox, eval_set=(X_val, y_val_boxcox), early_stopping_rounds=100)
    pred = model.predict(X_val)
    return rmsle_raw(y_val_raw, pred)


cb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
cb_study.optimize(cb_objective, n_trials=30, show_progress_bar=True)
print(f'Best CatBoost RMSLE (orig space): {cb_study.best_value:.6f}')
print(f'Best params: {cb_study.best_params}')

# ============================================================
# Save best params
# ============================================================
best_params = {
    'xgb': xgb_study.best_params,
    'lgb': lgb_study.best_params,
    'cb': cb_study.best_params,
    'xgb_rmsle': xgb_study.best_value,
    'lgb_rmsle': lgb_study.best_value,
    'cb_rmsle': cb_study.best_value,
}
with open('best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print('\n=== Tuning Summary (RMSLE in original SalePrice space) ===')
print(f'XGBoost:  {xgb_study.best_value:.6f}')
print(f'LightGBM: {lgb_study.best_value:.6f}')
print(f'CatBoost: {cb_study.best_value:.6f}')
