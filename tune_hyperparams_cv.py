"""Hyperparameter tuning with 5-fold CV. Practical: 30 trials per model, 1000 rounds."""
import numpy as np
import pandas as pd
import json
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from scipy.special import inv_boxcox
import optuna
import warnings
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
N_BOOST = 1000
EARLY_STOP = 50
bc_lambda = float(np.load('boxcox_lambda.npy'))

# Load existing best params from v1 as reference
try:
    with open('best_params.json') as f:
        prev_params = json.load(f)
except FileNotFoundError:
    prev_params = {}

folds = []
for i in range(N_FOLDS):
    X_tr = pd.read_csv(f'X_train_fold{i}.csv')
    X_val = pd.read_csv(f'X_val_fold{i}.csv')
    y_tr = np.load(f'y_train_fold{i}.npy')
    y_val = np.load(f'y_val_fold{i}.npy')
    folds.append((X_tr, X_val, y_tr, y_val))

feature_names = folds[0][0].columns.tolist()


def cv_rmsle(y_pred_list):
    rmsles = []
    for (_, _, _, y_val), y_pred in zip(folds, y_pred_list):
        y_pred_raw = inv_boxcox(y_pred, bc_lambda)
        y_pred_raw = np.maximum(y_pred_raw, 0)
        y_true_raw = inv_boxcox(y_val, bc_lambda)
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_true_raw), np.log1p(y_pred_raw)))
        rmsles.append(rmsle)
    return np.mean(rmsles)


# ============================================================
# XGBoost (30 trials)
# ============================================================
print('=== Tuning XGBoost (30 trials) ===')

prev_xgb = prev_params.get('xgb', {})
xgb_init = {
    'learning_rate': prev_xgb.get('learning_rate', 0.021),
    'max_depth': prev_xgb.get('max_depth', 3),
    'subsample': prev_xgb.get('subsample', 0.84),
    'colsample_bytree': prev_xgb.get('colsample_bytree', 0.66),
    'reg_alpha': prev_xgb.get('reg_alpha', 0.2),
    'reg_lambda': prev_xgb.get('reg_lambda', 0.032),
    'min_child_weight': prev_xgb.get('min_child_weight', 15),
}


def xgb_objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'seed': SEED,
    }
    fold_preds = []
    for X_tr, X_val, y_tr, y_val in folds:
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        bst = xgb.train(params, dtrain, num_boost_round=N_BOOST,
                        evals=[(dval, 'validation')],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        fold_preds.append(bst.predict(dval))
    return cv_rmsle(fold_preds)


# Seed with previous best params for faster convergence
xgb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
xgb_study.enqueue_trial(xgb_init)
xgb_study.optimize(xgb_objective, n_trials=30, show_progress_bar=True)
print(f'Best XGB CV RMSLE: {xgb_study.best_value:.6f}')
print(f'Best params: {xgb_study.best_params}')

# ============================================================
# LightGBM (30 trials)
# ============================================================
print('\n=== Tuning LightGBM (30 trials) ===')

prev_lgb = prev_params.get('lgb', {})
lgb_init = {
    'learning_rate': prev_lgb.get('learning_rate', 0.017),
    'max_depth': prev_lgb.get('max_depth', 3),
    'num_leaves': prev_lgb.get('num_leaves', 24),
    'subsample': prev_lgb.get('subsample', 0.77),
    'colsample_bytree': prev_lgb.get('colsample_bytree', 0.90),
    'reg_alpha': prev_lgb.get('reg_alpha', 0.014),
    'reg_lambda': prev_lgb.get('reg_lambda', 0.0012),
    'min_child_samples': prev_lgb.get('min_child_samples', 5),
}


def lgb_objective(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 8, 127),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'seed': SEED, 'verbose': -1,
    }
    fold_preds = []
    for X_tr, X_val, y_tr, y_val in folds:
        lgb_tr = lgb.Dataset(X_tr, label=y_tr)
        lgb_va = lgb.Dataset(X_val, label=y_val, reference=lgb_tr)
        bst = lgb.train(params, lgb_tr, num_boost_round=N_BOOST,
                        valid_sets=[lgb_va],
                        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)])
        fold_preds.append(bst.predict(X_val))
    return cv_rmsle(fold_preds)


lgb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
lgb_study.enqueue_trial(lgb_init)
lgb_study.optimize(lgb_objective, n_trials=30, show_progress_bar=True)
print(f'Best LGB CV RMSLE: {lgb_study.best_value:.6f}')
print(f'Best params: {lgb_study.best_params}')

# ============================================================
# CatBoost (30 trials)
# ============================================================
print('\n=== Tuning CatBoost (30 trials) ===')

prev_cb = prev_params.get('cb', {})
cb_init = {
    'learning_rate': prev_cb.get('learning_rate', 0.023),
    'depth': prev_cb.get('depth', 4),
    'subsample': prev_cb.get('subsample', 0.81),
    'l2_leaf_reg': prev_cb.get('l2_leaf_reg', 0.105),
    'random_strength': prev_cb.get('random_strength', 1.0),
    'border_count': 128,
}


def cb_objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'depth': trial.suggest_int('depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 5.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }
    fold_preds = []
    for X_tr, X_val, y_tr, y_val in folds:
        model = CatBoostRegressor(**params, iterations=N_BOOST, random_seed=SEED, verbose=False)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=EARLY_STOP)
        fold_preds.append(model.predict(X_val))
    return cv_rmsle(fold_preds)


cb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
cb_study.enqueue_trial(cb_init)
cb_study.optimize(cb_objective, n_trials=30, show_progress_bar=True)
print(f'Best CB CV RMSLE: {cb_study.best_value:.6f}')
print(f'Best params: {cb_study.best_params}')

# ============================================================
# Ridge (15 trials)
# ============================================================
print('\n=== Tuning Ridge (15 trials) ===')


def ridge_objective(trial):
    alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
    fold_preds = []
    for X_tr, X_val, y_tr, y_val in folds:
        model = Ridge(alpha=alpha, random_state=SEED)
        model.fit(X_tr, y_tr)
        fold_preds.append(model.predict(X_val))
    return cv_rmsle(fold_preds)


ridge_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
ridge_study.optimize(ridge_objective, n_trials=15, show_progress_bar=True)
print(f'Best Ridge CV RMSLE: {ridge_study.best_value:.6f}')
print(f'Best params: {ridge_study.best_params}')

# ============================================================
# Lasso (15 trials)
# ============================================================
print('\n=== Tuning Lasso (15 trials) ===')


def lasso_objective(trial):
    alpha = trial.suggest_float('alpha', 1e-5, 1.0, log=True)
    fold_preds = []
    for X_tr, X_val, y_tr, y_val in folds:
        model = Lasso(alpha=alpha, max_iter=5000, random_state=SEED)
        model.fit(X_tr, y_tr)
        fold_preds.append(model.predict(X_val))
    return cv_rmsle(fold_preds)


lasso_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
lasso_study.optimize(lasso_objective, n_trials=15, show_progress_bar=True)
print(f'Best Lasso CV RMSLE: {lasso_study.best_value:.6f}')
print(f'Best params: {lasso_study.best_params}')

# ============================================================
# ElasticNet (15 trials)
# ============================================================
print('\n=== Tuning ElasticNet (15 trials) ===')


def enet_objective(trial):
    alpha = trial.suggest_float('alpha', 1e-5, 1.0, log=True)
    l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
    fold_preds = []
    for X_tr, X_val, y_tr, y_val in folds:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=SEED)
        model.fit(X_tr, y_tr)
        fold_preds.append(model.predict(X_val))
    return cv_rmsle(fold_preds)


enet_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
enet_study.optimize(enet_objective, n_trials=15, show_progress_bar=True)
print(f'Best EN CV RMSLE: {enet_study.best_value:.6f}')
print(f'Best params: {enet_study.best_params}')

# ============================================================
# Save
# ============================================================
best_params = {
    'xgb': xgb_study.best_params,
    'lgb': lgb_study.best_params,
    'cb': cb_study.best_params,
    'ridge': ridge_study.best_params,
    'lasso': lasso_study.best_params,
    'enet': enet_study.best_params,
    'xgb_rmsle': xgb_study.best_value,
    'lgb_rmsle': lgb_study.best_value,
    'cb_rmsle': cb_study.best_value,
    'ridge_rmsle': ridge_study.best_value,
    'lasso_rmsle': lasso_study.best_value,
    'enet_rmsle': enet_study.best_value,
}
with open('best_params_v2.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print('\n=== Tuning Summary (CV RMSLE) ===')
print(f'XGBoost:     {xgb_study.best_value:.6f}')
print(f'LightGBM:    {lgb_study.best_value:.6f}')
print(f'CatBoost:    {cb_study.best_value:.6f}')
print(f'Ridge:       {ridge_study.best_value:.6f}')
print(f'Lasso:       {lasso_study.best_value:.6f}')
print(f'ElasticNet:  {enet_study.best_value:.6f}')
