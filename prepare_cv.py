"""5-fold CV data preparation with nested target encoding and per-fold scaling."""
import pandas as pd
import numpy as np
import joblib
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

SEED = 42
N_OUTER = 5
N_INNER = 10
TARGET_ENCODE_COLS = ['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SaleType', 'MSZoning']
LOG_FEATURES = ['GrLivArea', 'TotalBsmtSF', 'LotArea', 'TotalSF']

# Load data
fn = 'train_featured_v2.csv'
df = pd.read_csv(fn)
print(f'Loaded: {len(df)} rows')

if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# ---- Fill Missing Values ----
# LotFrontage: use neighborhood median (same as predict_submission_v2.py)
if 'LotFrontage' in df.columns:
    median_by_nbhd = df.groupby('Neighborhood')['LotFrontage'].median()
    df['LotFrontage'] = df.apply(
        lambda row: median_by_nbhd.get(row['Neighborhood'], np.nan)
        if pd.isna(row['LotFrontage']) else row['LotFrontage'],
        axis=1,
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# All other numeric/categorical NaNs
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
n_removed = mask_outlier.sum()
df = df[~mask_outlier].reset_index(drop=True)
print(f'Removed {n_removed} outliers, {len(df)} rows remaining')

# ---- Box-Cox Transform ----
y_raw = df['SalePrice'].values
y, bc_lambda = stats.boxcox(y_raw)
np.save('boxcox_lambda.npy', bc_lambda)
print(f'Box-Cox lambda: {bc_lambda:.6f}')

# ---- Log-transform ----
for col in LOG_FEATURES:
    if col in df.columns:
        df[col] = np.log1p(df[col])

# ---- Prepare full-data target encoding (for final submission artifacts) ----
X_full_raw = df.drop(columns=['SalePrice'])

# Full-data 10-fold target encoding
full_te_means = {}
df_te = X_full_raw.copy()
for col in TARGET_ENCODE_COLS:
    if col not in df_te.columns:
        continue
    te_col = f'{col}_TE'
    df_te[te_col] = np.nan
    kf_inner = KFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
    for tr_idx, val_idx in kf_inner.split(df_te):
        fold_mean = pd.Series(y[tr_idx]).groupby(df_te[col].iloc[tr_idx].values).mean()
        df_te.loc[val_idx, te_col] = df_te.loc[val_idx, col].map(fold_mean)
    df_te[te_col] = df_te[te_col].fillna(y.mean())
    full_te_means[col] = pd.Series(y).groupby(df_te[col]).mean().to_dict()

# Full-data scaler
num_cols = df_te.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_te.select_dtypes(exclude=[np.number]).columns.tolist()

scaler_all = StandardScaler()
X_num_all = pd.DataFrame(scaler_all.fit_transform(df_te[num_cols]), columns=num_cols, index=df_te.index)
if cat_cols:
    X_cat_all = pd.get_dummies(df_te[cat_cols], drop_first=False)
    X_all_processed = pd.concat([X_num_all, X_cat_all], axis=1)
else:
    X_all_processed = X_num_all

joblib.dump(scaler_all, 'scaler_all.pkl')
np.save('train_columns_all.npy', X_all_processed.columns.tolist())
np.save('target_encode_means.npy', full_te_means, allow_pickle=True)
print(f'Full-data shape: {X_all_processed.shape}')

# ---- 5-fold CV with per-fold preprocessing ----
outer_kf = KFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)

all_indices = np.arange(len(df))

for fold_idx, (train_idx, val_idx) in enumerate(outer_kf.split(all_indices)):
    print(f'\n=== Fold {fold_idx + 1}/{N_OUTER} ===')

    X_fold = df_te.copy()
    y_fold = y.copy()

    # Split
    X_tr_raw = X_fold.iloc[train_idx].reset_index(drop=True)
    X_val_raw = X_fold.iloc[val_idx].reset_index(drop=True)
    y_tr = y_fold[train_idx]
    y_val = y_fold[val_idx]

    # ---- Nested 10-fold target encoding for training set ----
    te_fold_means = {}
    for col in TARGET_ENCODE_COLS:
        if col not in X_tr_raw.columns:
            continue
        te_col = f'{col}_TE'
        X_tr_raw[te_col] = np.nan
        # Drop existing TE column from validation raw if present
        if te_col in X_val_raw.columns:
            X_val_raw = X_val_raw.drop(columns=[te_col])

        inner_kf = KFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
        for inner_tr, inner_val in inner_kf.split(X_tr_raw):
            inner_mean = pd.Series(y_tr[inner_tr]).groupby(
                X_tr_raw[col].iloc[inner_tr].values
            ).mean()
            X_tr_raw.loc[inner_val, te_col] = X_tr_raw.loc[inner_val, col].map(inner_mean)

        X_tr_raw[te_col] = X_tr_raw[te_col].fillna(y_tr.mean())
        # Store train-fold means for validation set
        te_fold_means[col] = X_tr_raw.groupby(col)[te_col].first().to_dict()
        # Apply to validation
        X_val_raw[te_col] = X_val_raw[col].map(te_fold_means[col])
        X_val_raw[te_col] = X_val_raw[te_col].fillna(y_tr.mean())

    # ---- Per-fold StandardScaler ----
    num_cols_tr = X_tr_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_tr = X_tr_raw.select_dtypes(exclude=[np.number]).columns.tolist()

    scaler_fold = StandardScaler()
    X_tr_num = pd.DataFrame(
        scaler_fold.fit_transform(X_tr_raw[num_cols_tr]),
        columns=num_cols_tr, index=X_tr_raw.index
    )
    X_val_num = pd.DataFrame(
        scaler_fold.transform(X_val_raw[num_cols_tr]),
        columns=num_cols_tr, index=X_val_raw.index
    )

    if cat_cols_tr:
        X_tr_cat = pd.get_dummies(X_tr_raw[cat_cols_tr], drop_first=False)
        X_val_cat = pd.get_dummies(X_val_raw[cat_cols_tr], drop_first=False)
        X_tr = pd.concat([X_tr_num, X_tr_cat], axis=1)
        X_val = pd.concat([X_val_num, X_val_cat], axis=1)
        # Align columns
        all_cols = X_tr.columns.union(X_val.columns)
        X_tr = X_tr.reindex(columns=all_cols, fill_value=0)
        X_val = X_val.reindex(columns=all_cols, fill_value=0)
    else:
        X_tr = X_tr_num
        X_val = X_val_num

    print(f'  Train: {X_tr.shape}, Val: {X_val.shape}')

    # Save fold data
    X_tr.to_csv(f'X_train_fold{fold_idx}.csv', index=False)
    X_val.to_csv(f'X_val_fold{fold_idx}.csv', index=False)
    np.save(f'y_train_fold{fold_idx}.npy', y_tr)
    np.save(f'y_val_fold{fold_idx}.npy', y_val)

print('\n=== CV data preparation complete ===')
print(f'Full processed data: {X_all_processed.shape}')
print(f'X_all_processed columns saved to train_columns_all.npy')
print('Saved: scaler_all.pkl, target_encode_means.npy, boxcox_lambda.npy')
for i in range(N_OUTER):
    print(f'Fold {i}: X_train_fold{i}.csv, X_val_fold{i}.csv, y_train_fold{i}.npy, y_val_fold{i}.npy')
