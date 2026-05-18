"""Optuna hyperparameter tuning for XGBoost, LightGBM, CatBoost.
Saves best params to disk for stacking ensemble."""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import optuna
import json

SEED = 42
N_TRIALS = 100

# Load data
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
feature_names = X_train.columns.tolist()


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ============================================================
# XGBoost Tuning
# ============================================================
def xgb_objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 20),
        'seed': SEED,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    bst = xgb.train(
        params, dtrain, num_boost_round=2000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=100, verbose_eval=False,
    )
    pred = bst.predict(dval)
    return rmsle(y_val, pred)


print('=== Tuning XGBoost (100 trials) ===')
xgb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f'Best XGBoost RMSLE: {xgb_study.best_value:.6f}')
print(f'Best XGBoost params: {xgb_study.best_params}')

# ============================================================
# LightGBM Tuning
# ============================================================
def lgb_objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 8, 127),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'seed': SEED,
        'verbose': -1,
    }
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    bst = lgb.train(
        params, lgb_train, num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    pred = bst.predict(X_val)
    return rmsle(y_val, pred)


print('\n=== Tuning LightGBM (100 trials) ===')
lgb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
lgb_study.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f'Best LightGBM RMSLE: {lgb_study.best_value:.6f}')
print(f'Best LightGBM params: {lgb_study.best_params}')

# ============================================================
# CatBoost Tuning
# ============================================================
def cb_objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
    }
    model = CatBoostRegressor(
        **params, iterations=2000, random_seed=SEED, verbose=False,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    pred = model.predict(X_val)
    return rmsle(y_val, pred)


print('\n=== Tuning CatBoost (100 trials) ===')
cb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
cb_study.optimize(cb_objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f'Best CatBoost RMSLE: {cb_study.best_value:.6f}')
print(f'Best CatBoost params: {cb_study.best_params}')

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

print('\n=== Summary ===')
print(f'XGBoost:  {xgb_study.best_value:.6f} (params saved)')
print(f'LightGBM: {lgb_study.best_value:.6f} (params saved)')
print(f'CatBoost: {cb_study.best_value:.6f} (params saved)')
print('\nBest params saved to best_params.json')
