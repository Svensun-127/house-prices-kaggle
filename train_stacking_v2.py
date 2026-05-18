"""Stacking ensemble v2: 7 base models + Ridge meta-model.
Reads train_featured_v2.csv ONCE (each sample appears once).
5-fold CV with per-fold preprocessing (nested TE, scaling, one-hot).
Then retrains base models on full data using saved scaler_all/te_means/train_columns_all.
"""
import numpy as np
import pandas as pd
import joblib
import json
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.special import inv_boxcox
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
N_TE_INNER = 10
TARGET_ENCODE_COLS = ['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SaleType', 'MSZoning']
LOG_FEATURES = ['GrLivArea', 'TotalBsmtSF', 'LotArea', 'TotalSF']

bc_lambda = float(np.load('boxcox_lambda.npy'))

# Load tuned params
with open('best_params_v2.json') as f:
    tuned = json.load(f)

# ============================================================
# 1. Load raw featured data
# ============================================================
print('=== Loading data ===')
df = pd.read_csv('train_featured_v2.csv')

if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# ---- Fill Missing Values (same as prepare_cv.py) ----
if 'LotFrontage' in df.columns:
    median_by_nbhd = df.groupby('Neighborhood')['LotFrontage'].median()
    df['LotFrontage'] = df.apply(
        lambda row: median_by_nbhd.get(row['Neighborhood'], np.nan)
        if pd.isna(row['LotFrontage']) else row['LotFrontage'],
        axis=1,
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

for col in df.columns:
    if col == 'LotFrontage':
        continue
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna('None')

# ---- Outlier Removal ----
mask_outlier = (
    ((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)) |
    (df['LotArea'] > 100000) |
    (df['TotalBsmtSF'] > 3000)
)
df = df[~mask_outlier].reset_index(drop=True)
print(f'{len(df)} rows after outlier removal')

# ---- Box-Cox Transform ----
y_raw = df['SalePrice'].values
y = stats.boxcox(y_raw, lmbda=bc_lambda)
y = y.astype(np.float64)

# ---- Log-transform ----
for col in LOG_FEATURES:
    if col in df.columns:
        df[col] = np.log1p(df[col])

# Raw dataframe (before TE/scaling) for per-fold CV
X_raw = df.drop(columns=['SalePrice'])
num_cols_raw = X_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_raw = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
all_indices = np.arange(len(df))
print(f'Features: {len(num_cols_raw)} numeric + {len(cat_cols_raw)} categorical = {X_raw.shape[1]} total')


def rmsle_raw(y_true_boxcox, y_pred_boxcox):
    y_true_raw = inv_boxcox(y_true_boxcox, bc_lambda)
    y_pred_raw = inv_boxcox(y_pred_boxcox, bc_lambda)
    y_pred_raw = np.maximum(y_pred_raw, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true_raw), np.log1p(y_pred_raw)))


# ============================================================
# 2. 5-fold CV to generate meta-features (per-fold preprocessing)
# ============================================================
print('\n=== Generating meta-features with 5-fold CV ===')

N_BASE = 7
meta_train = np.zeros((len(df), N_BASE))

outer_kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold_idx, (tr_idx, val_idx) in enumerate(outer_kf.split(all_indices)):
    print(f'\nFold {fold_idx + 1}/{N_FOLDS}')
    y_tr, y_val_f = y[tr_idx], y[val_idx]

    # ---- Per-fold target encoding (nested 10-fold on training) ----
    X_tr_te = X_raw.iloc[tr_idx].reset_index(drop=True)
    X_val_te = X_raw.iloc[val_idx].reset_index(drop=True)

    for col in TARGET_ENCODE_COLS:
        if col not in X_tr_te.columns:
            continue
        te_col = f'{col}_TE'
        X_tr_te[te_col] = np.nan

        inner_kf = KFold(n_splits=N_TE_INNER, shuffle=True, random_state=SEED)
        for inner_tr, inner_val in inner_kf.split(X_tr_te):
            inner_mean = pd.Series(y_tr[inner_tr]).groupby(
                X_tr_te[col].iloc[inner_tr].values
            ).mean()
            X_tr_te.loc[inner_val, te_col] = X_tr_te.loc[inner_val, col].map(inner_mean)

        X_tr_te[te_col] = X_tr_te[te_col].fillna(y_tr.mean())
        fold_mean = X_tr_te.groupby(col)[te_col].first().to_dict()
        X_val_te[te_col] = X_val_te[col].map(fold_mean)
        X_val_te[te_col] = X_val_te[te_col].fillna(y_tr.mean())

    # ---- Per-fold scaling + one-hot ----
    num_cols = X_tr_te.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_tr_te.select_dtypes(exclude=[np.number]).columns.tolist()

    scaler_fold = StandardScaler()
    X_tr_num = pd.DataFrame(scaler_fold.fit_transform(X_tr_te[num_cols]),
                            columns=num_cols, index=X_tr_te.index)
    X_val_num = pd.DataFrame(scaler_fold.transform(X_val_te[num_cols]),
                             columns=num_cols, index=X_val_te.index)

    if cat_cols:
        X_tr_cat = pd.get_dummies(X_tr_te[cat_cols], drop_first=False)
        X_val_cat = pd.get_dummies(X_val_te[cat_cols], drop_first=False)
        X_tr = pd.concat([X_tr_num, X_tr_cat], axis=1)
        X_val_f = pd.concat([X_val_num, X_val_cat], axis=1)
        all_cols_fold = X_tr.columns.union(X_val_f.columns)
        X_tr = X_tr.reindex(columns=all_cols_fold, fill_value=0)
        X_val_f = X_val_f.reindex(columns=all_cols_fold, fill_value=0)
    else:
        X_tr = X_tr_num
        X_val_f = X_val_num

    fold_feature_names = X_tr.columns.tolist()

    # ---- Train 7 base models ----
    # XGBoost
    xgb_params = {
        'objective': 'reg:squarederror',
        'learning_rate': tuned['xgb']['learning_rate'],
        'max_depth': int(tuned['xgb']['max_depth']),
        'subsample': tuned['xgb']['subsample'],
        'colsample_bytree': tuned['xgb']['colsample_bytree'],
        'reg_alpha': tuned['xgb']['reg_alpha'],
        'reg_lambda': tuned['xgb']['reg_lambda'],
        'min_child_weight': int(tuned['xgb']['min_child_weight']),
        'seed': SEED,
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=fold_feature_names)
    dval = xgb.DMatrix(X_val_f, label=y_val_f, feature_names=fold_feature_names)
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=1000,
                          evals=[(dval, 'validation')],
                          early_stopping_rounds=100, verbose_eval=False)
    meta_train[val_idx, 0] = xgb_model.predict(dval)
    print(f'  XGB:  {rmsle_raw(y_val_f, meta_train[val_idx, 0]):.6f}')

    # LightGBM
    lgb_model = lgb.train(
        {'objective': 'regression', 'metric': 'rmse',
         'learning_rate': tuned['lgb']['learning_rate'],
         'max_depth': int(tuned['lgb']['max_depth']),
         'num_leaves': int(tuned['lgb']['num_leaves']),
         'subsample': tuned['lgb']['subsample'],
         'colsample_bytree': tuned['lgb']['colsample_bytree'],
         'reg_alpha': tuned['lgb']['reg_alpha'],
         'reg_lambda': tuned['lgb']['reg_lambda'],
         'min_child_samples': int(tuned['lgb']['min_child_samples']),
         'seed': SEED, 'verbose': -1},
        lgb.Dataset(X_tr, label=y_tr), num_boost_round=1000,
        valid_sets=[lgb.Dataset(X_val_f, label=y_val_f)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    meta_train[val_idx, 1] = lgb_model.predict(X_val_f)
    print(f'  LGB:  {rmsle_raw(y_val_f, meta_train[val_idx, 1]):.6f}')

    # CatBoost
    cb_model = CatBoostRegressor(
        learning_rate=tuned['cb']['learning_rate'],
        depth=int(tuned['cb']['depth']),
        subsample=tuned['cb']['subsample'],
        l2_leaf_reg=tuned['cb']['l2_leaf_reg'],
        random_strength=tuned['cb']['random_strength'],
        border_count=int(tuned['cb'].get('border_count', 128)),
        iterations=1000, random_seed=SEED, verbose=False,
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_val_f, y_val_f), early_stopping_rounds=100)
    meta_train[val_idx, 2] = cb_model.predict(X_val_f)
    print(f'  CB:   {rmsle_raw(y_val_f, meta_train[val_idx, 2]):.6f}')

    # RandomForest
    rf_model = RandomForestRegressor(
        n_estimators=500, max_depth=12, min_samples_leaf=4,
        random_state=SEED, n_jobs=-1,
    )
    rf_model.fit(X_tr, y_tr)
    meta_train[val_idx, 3] = rf_model.predict(X_val_f)
    print(f'  RF:   {rmsle_raw(y_val_f, meta_train[val_idx, 3]):.6f}')

    # Ridge
    ridge_model = Ridge(alpha=tuned['ridge']['alpha'], random_state=SEED)
    ridge_model.fit(X_tr, y_tr)
    meta_train[val_idx, 4] = ridge_model.predict(X_val_f)
    print(f'  Ridge: {rmsle_raw(y_val_f, meta_train[val_idx, 4]):.6f}')

    # Lasso
    lasso_model = Lasso(alpha=tuned['lasso']['alpha'], max_iter=5000, random_state=SEED)
    lasso_model.fit(X_tr, y_tr)
    meta_train[val_idx, 5] = lasso_model.predict(X_val_f)
    print(f'  Lasso: {rmsle_raw(y_val_f, meta_train[val_idx, 5]):.6f}')

    # ElasticNet
    enet_model = ElasticNet(alpha=tuned['enet']['alpha'], l1_ratio=tuned['enet']['l1_ratio'],
                            max_iter=5000, random_state=SEED)
    enet_model.fit(X_tr, y_tr)
    meta_train[val_idx, 6] = enet_model.predict(X_val_f)
    print(f'  EN:   {rmsle_raw(y_val_f, meta_train[val_idx, 6]):.6f}')

# OOF RMSLE per base model
print('\n=== OOF RMSLE (original space) ===')
model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'Ridge', 'Lasso', 'ElasticNet']
for i, name in enumerate(model_names):
    score = rmsle_raw(y, meta_train[:, i])
    print(f'  {name:<12s}: {score:.6f}')

# ============================================================
# 3. Train meta-models
# ============================================================
print('\n=== Training meta-models ===')

meta_ridge = Ridge(alpha=1.0, random_state=SEED)
meta_ridge.fit(meta_train, y)
ridge_pred = meta_ridge.predict(meta_train)
ridge_oof = rmsle_raw(y, ridge_pred)
print(f'Meta Ridge OOF RMSLE:     {ridge_oof:.6f}')
print(f'Coefs: {np.round(meta_ridge.coef_, 4)}')

meta_enet = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=SEED)
meta_enet.fit(meta_train, y)
enet_pred = meta_enet.predict(meta_train)
enet_oof = rmsle_raw(y, enet_pred)
print(f'Meta ElasticNet OOF RMSLE: {enet_oof:.6f}')
print(f'Coefs: {np.round(meta_enet.coef_, 4)}')

if ridge_oof <= enet_oof:
    meta_model = meta_ridge
    print(f'Selected: Ridge (OOF: {ridge_oof:.6f})')
else:
    meta_model = meta_enet
    print(f'Selected: ElasticNet (OOF: {enet_oof:.6f})')

# ============================================================
# 4. Preprocess full data using saved artifacts (matches predict_submission_v2.py)
# ============================================================
print('\n=== Preprocessing full data with saved artifacts ===')

scaler_all = joblib.load('scaler_all.pkl')
train_columns_all = np.load('train_columns_all.npy', allow_pickle=True).tolist()
te_means = np.load('target_encode_means.npy', allow_pickle=True).item()

# Apply target encoding
X_fe = X_raw.copy()
for col, mapping in te_means.items():
    if col in X_fe.columns:
        te_col = f'{col}_TE'
        X_fe[te_col] = X_fe[col].map(mapping)
        mean_val = np.mean(list(mapping.values())) if mapping else 0
        X_fe[te_col] = X_fe[te_col].fillna(mean_val)

# Scale + one-hot
num_cols = X_fe.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_fe.select_dtypes(exclude=[np.number]).columns.tolist()
X_num = pd.DataFrame(scaler_all.transform(X_fe[num_cols]), columns=num_cols, index=X_fe.index)
if cat_cols:
    X_cat = pd.get_dummies(X_fe[cat_cols], drop_first=False)
    X_all = pd.concat([X_num, X_cat], axis=1)
else:
    X_all = X_num
X_all = X_all.reindex(columns=train_columns_all, fill_value=0)
print(f'Full processed shape: {X_all.shape}')

# ============================================================
# 5. Retrain base models on FULL data
# ============================================================
print('\n=== Retraining base models on full data ===')

feature_names = train_columns_all

# XGBoost
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': tuned['xgb']['learning_rate'],
    'max_depth': int(tuned['xgb']['max_depth']),
    'subsample': tuned['xgb']['subsample'],
    'colsample_bytree': tuned['xgb']['colsample_bytree'],
    'reg_alpha': tuned['xgb']['reg_alpha'],
    'reg_lambda': tuned['xgb']['reg_lambda'],
    'min_child_weight': int(tuned['xgb']['min_child_weight']),
    'seed': SEED,
}
dtrain_full = xgb.DMatrix(X_all, label=y, feature_names=feature_names)
xgb_final = xgb.train(xgb_params, dtrain_full, num_boost_round=1000, verbose_eval=False)
xgb_final.save_model('xgb_stacking_v2.json')
print('XGBoost saved.')

# LightGBM
lgb_final = lgb.train(
    {'objective': 'regression', 'metric': 'rmse',
     'learning_rate': tuned['lgb']['learning_rate'],
     'max_depth': int(tuned['lgb']['max_depth']),
     'num_leaves': int(tuned['lgb']['num_leaves']),
     'subsample': tuned['lgb']['subsample'],
     'colsample_bytree': tuned['lgb']['colsample_bytree'],
     'reg_alpha': tuned['lgb']['reg_alpha'],
     'reg_lambda': tuned['lgb']['reg_lambda'],
     'min_child_samples': int(tuned['lgb']['min_child_samples']),
     'seed': SEED, 'verbose': -1},
    lgb.Dataset(X_all, label=y), num_boost_round=1000,
)
lgb_final.save_model('lgb_stacking_v2.txt')
print('LightGBM saved.')

# CatBoost
cb_final = CatBoostRegressor(
    learning_rate=tuned['cb']['learning_rate'],
    depth=int(tuned['cb']['depth']),
    subsample=tuned['cb']['subsample'],
    l2_leaf_reg=tuned['cb']['l2_leaf_reg'],
    random_strength=tuned['cb']['random_strength'],
    border_count=int(tuned['cb'].get('border_count', 128)),
    iterations=1000, random_seed=SEED, verbose=False,
)
cb_final.fit(X_all, y)
cb_final.save_model('cb_stacking_v2.cbm')
print('CatBoost saved.')

# RF
rf_final = RandomForestRegressor(
    n_estimators=500, max_depth=12, min_samples_leaf=4,
    random_state=SEED, n_jobs=-1,
)
rf_final.fit(X_all, y)
joblib.dump(rf_final, 'rf_stacking_v2.pkl')
print('RandomForest saved.')

# Ridge
ridge_final = Ridge(alpha=tuned['ridge']['alpha'], random_state=SEED)
ridge_final.fit(X_all, y)
joblib.dump(ridge_final, 'ridge_stacking_v2.pkl')
print('Ridge saved.')

# Lasso
lasso_final = Lasso(alpha=tuned['lasso']['alpha'], max_iter=5000, random_state=SEED)
lasso_final.fit(X_all, y)
joblib.dump(lasso_final, 'lasso_stacking_v2.pkl')
print('Lasso saved.')

# ElasticNet
enet_final = ElasticNet(alpha=tuned['enet']['alpha'], l1_ratio=tuned['enet']['l1_ratio'],
                        max_iter=5000, random_state=SEED)
enet_final.fit(X_all, y)
joblib.dump(enet_final, 'enet_stacking_v2.pkl')
print('ElasticNet saved.')

# Save meta model
joblib.dump(meta_model, 'meta_model_v2.pkl')
print('Meta-model saved.')

np.save('feature_names_v2.npy', feature_names)

print('\n=== Done ===')
print(f'Meta-model OOF: {min(ridge_oof, enet_oof):.6f}')
