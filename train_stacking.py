"""Stacking Ensemble: XGB + LGB + CB + RF base models, Ridge meta-model.
Uses 5-fold CV to generate meta-features. Tuned params loaded from best_params.json."""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import json
import os
import warnings
warnings.filterwarnings('ignore')

SEED = 42


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Load data
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
feature_names = X_train.columns.tolist()

# Load best params from Optuna (or use defaults)
if os.path.exists('best_params.json'):
    with open('best_params.json') as f:
        best = json.load(f)
    xgb_best = best['xgb']
    lgb_best = best['lgb']
    cb_best = best['cb']
    print('Loaded tuned params from best_params.json')
else:
    xgb_best = {'learning_rate': 0.03, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_weight': 1}
    lgb_best = {'learning_rate': 0.03, 'max_depth': 4, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 20}
    cb_best = {'learning_rate': 0.03, 'depth': 4, 'subsample': 0.8, 'l2_leaf_reg': 1.0, 'random_strength': 1.0}
    print('Using default params (best_params.json not found)')

# Combine train+val for 5-fold stacking
X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_full = np.concatenate([y_train, y_val])

n_base = 4  # XGB, LGB, CB, RF
n_samples = len(X_full)
meta_features = np.zeros((n_samples, n_base))

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

print(f'\n=== 5-Fold Stacking: {n_samples} samples, {n_base} base models ===')

# Store OOF predictions per model for individual scoring
oof_preds = np.zeros((n_samples, n_base))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
    X_tr, X_va = X_full.iloc[train_idx], X_full.iloc[val_idx]
    y_tr, y_va = y_full[train_idx], y_full[val_idx]
    print(f'\nFold {fold+1}/5: train={len(X_tr)}, val={len(X_va)}')

    # --- XGBoost ---
    xgb_params_fold = {
        'objective': 'reg:squarederror',
        'learning_rate': xgb_best.get('learning_rate', 0.03),
        'max_depth': int(xgb_best.get('max_depth', 4)),
        'subsample': xgb_best.get('subsample', 0.8),
        'colsample_bytree': xgb_best.get('colsample_bytree', 0.8),
        'reg_alpha': xgb_best.get('reg_alpha', 0.1),
        'reg_lambda': xgb_best.get('reg_lambda', 1.0),
        'min_child_weight': xgb_best.get('min_child_weight', 1),
        'seed': SEED,
    }
    dtrain_fold = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dval_fold = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
    xgb_bst = xgb.train(
        xgb_params_fold, dtrain_fold, num_boost_round=2000,
        evals=[(dval_fold, 'validation')],
        early_stopping_rounds=100, verbose_eval=False,
    )
    meta_features[val_idx, 0] = xgb_bst.predict(dval_fold)
    oof_preds[val_idx, 0] = meta_features[val_idx, 0]

    # --- LightGBM ---
    lgb_params_fold = {
        'objective': 'regression', 'metric': 'rmse',
        'learning_rate': lgb_best.get('learning_rate', 0.03),
        'max_depth': int(lgb_best.get('max_depth', 4)),
        'num_leaves': int(lgb_best.get('num_leaves', 31)),
        'subsample': lgb_best.get('subsample', 0.8),
        'colsample_bytree': lgb_best.get('colsample_bytree', 0.8),
        'reg_alpha': lgb_best.get('reg_alpha', 0.1),
        'reg_lambda': lgb_best.get('reg_lambda', 1.0),
        'min_child_samples': int(lgb_best.get('min_child_samples', 20)),
        'seed': SEED, 'verbose': -1,
    }
    lgb_tr = lgb.Dataset(X_tr, label=y_tr)
    lgb_va = lgb.Dataset(X_va, label=y_va, reference=lgb_tr)
    lgb_bst = lgb.train(
        lgb_params_fold, lgb_tr, num_boost_round=2000,
        valid_sets=[lgb_va],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    meta_features[val_idx, 1] = lgb_bst.predict(X_va)
    oof_preds[val_idx, 1] = meta_features[val_idx, 1]

    # --- CatBoost ---
    cb_bst = CatBoostRegressor(
        learning_rate=cb_best.get('learning_rate', 0.03),
        depth=int(cb_best.get('depth', 4)),
        subsample=cb_best.get('subsample', 0.8),
        l2_leaf_reg=cb_best.get('l2_leaf_reg', 1.0),
        random_strength=cb_best.get('random_strength', 1.0),
        iterations=2000, random_seed=SEED, verbose=False,
    )
    cb_bst.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)
    meta_features[val_idx, 2] = cb_bst.predict(X_va)
    oof_preds[val_idx, 2] = meta_features[val_idx, 2]

    # --- Random Forest ---
    rf_bst = RandomForestRegressor(
        n_estimators=500, max_depth=10, min_samples_leaf=5,
        random_state=SEED, n_jobs=-1,
    )
    rf_bst.fit(X_tr, y_tr)
    meta_features[val_idx, 3] = rf_bst.predict(X_va)
    oof_preds[val_idx, 3] = meta_features[val_idx, 3]

# ---- Individual model OOF scores ----
print('\n=== Individual Model OOF RMSLE ===')
model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']
for i, name in enumerate(model_names):
    score = rmsle(y_full, oof_preds[:, i])
    print(f'  {name}: {score:.6f}')

# ---- Simple average of base models ----
avg_pred = oof_preds.mean(axis=1)
avg_rmsle = rmsle(y_full, avg_pred)
print(f'\nSimple average OOF RMSLE: {avg_rmsle:.6f}')

# ---- Train meta-model (Ridge) ----
print('\n=== Meta-Model: Ridge ===')
meta_ridge = Ridge(alpha=1.0, random_state=SEED)
meta_ridge.fit(meta_features, y_full)
ridge_pred = meta_ridge.predict(meta_features)
ridge_rmsle = rmsle(y_full, ridge_pred)
print(f'Ridge stacking OOF RMSLE: {ridge_rmsle:.6f}')
print(f'Ridge coefficients: {dict(zip(model_names, meta_ridge.coef_))}')

# ---- Try ElasticNet ----
print('\n=== Meta-Model: ElasticNet ===')
meta_en = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=SEED)
meta_en.fit(meta_features, y_full)
en_pred = meta_en.predict(meta_features)
en_rmsle = rmsle(y_full, en_pred)
print(f'ElasticNet stacking OOF RMSLE: {en_rmsle:.6f}')
print(f'ElasticNet coefficients: {dict(zip(model_names, meta_en.coef_))}')

# ---- Best meta-model ----
best_meta_name = 'Ridge' if ridge_rmsle <= en_rmsle else 'ElasticNet'
best_meta = meta_ridge if ridge_rmsle <= en_rmsle else meta_en
best_oof = min(ridge_rmsle, en_rmsle)
print(f'\nBest meta-model: {best_meta_name} (OOF RMSLE: {best_oof:.6f})')

# ---- Train full base models on all data for test prediction ----
print('\n=== Training final base models on full data ===')

# XGBoost full
dtrain_full = xgb.DMatrix(X_full, label=y_full, feature_names=feature_names)
xgb_final = xgb.train(xgb_params_fold, dtrain_full, num_boost_round=500, verbose_eval=False)

# LightGBM full
lgb_full_ds = lgb.Dataset(X_full, label=y_full)
lgb_final = lgb.train(lgb_params_fold, lgb_full_ds, num_boost_round=500)

# CatBoost full
cb_final = CatBoostRegressor(
    learning_rate=cb_best.get('learning_rate', 0.03),
    depth=int(cb_best.get('depth', 4)),
    subsample=cb_best.get('subsample', 0.8),
    l2_leaf_reg=cb_best.get('l2_leaf_reg', 1.0),
    random_strength=cb_best.get('random_strength', 1.0),
    iterations=500, random_seed=SEED, verbose=False,
)
cb_final.fit(X_full, y_full)

# Random Forest full
rf_final = RandomForestRegressor(
    n_estimators=500, max_depth=10, min_samples_leaf=5,
    random_state=SEED, n_jobs=-1,
)
rf_final.fit(X_full, y_full)

# ---- Save models ----
xgb_final.save_model('xgb_stacking_model.json')
lgb_final.save_model('lgb_stacking_model.txt')
cb_final.save_model('cb_stacking_model.cbm')
np.save('rf_stacking_model.npy', rf_final, allow_pickle=True)
np.save('meta_model.npy', best_meta, allow_pickle=True)
np.save('feature_names.npy', feature_names, allow_pickle=True)

print('\n=== Models saved ===')
print('  xgb_stacking_model.json')
print('  lgb_stacking_model.txt')
print('  cb_stacking_model.cbm')
print('  rf_stacking_model.npy')
print('  meta_model.npy')

# ---- Final summary ----
print(f'\n=== Final Summary ===')
print(f'Best individual OOF: {min(*[rmsle(y_full, oof_preds[:, i]) for i in range(4)]):.6f}')
print(f'Stacking OOF: {best_oof:.6f}')
print(f'Meta-model: {best_meta_name}')
