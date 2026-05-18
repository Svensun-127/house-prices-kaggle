import pandas as pd
import numpy as np
import joblib
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

# Load featured data
fn = 'train_featured.csv'
df = pd.read_csv(fn)
print(f'Loaded: {len(df)} rows')

# Step 1: drop Id
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# ---- Outlier Removal ----
print(f'\nBefore outlier removal: {len(df)} rows')

mask_outlier = (
    ((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)) |
    (df['LotArea'] > 100000) |
    (df['TotalBsmtSF'] > 3000)
)
# Count outliers before removal
n_grliv = ((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)).sum()
n_lot = (df['LotArea'] > 100000).sum()
n_bsmt = (df['TotalBsmtSF'] > 3000).sum()
mask_outlier = (
    ((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)) |
    (df['LotArea'] > 100000) |
    (df['TotalBsmtSF'] > 3000)
)
n_removed = mask_outlier.sum()
df = df[~mask_outlier].reset_index(drop=True)
print(f'Removed {n_removed} outliers:')
print(f'  - GrLivArea>4000 & SalePrice<300k: {n_grliv}')
print(f'  - LotArea>100000: {n_lot}')
print(f'  - TotalBsmtSF>3000: {n_bsmt}')
print(f'  After removal: {len(df)} rows')

# ---- Box-Cox Transform on SalePrice ----
y_raw = df['SalePrice'].values
y, bc_lambda = stats.boxcox(y_raw)
print(f'\nBox-Cox lambda: {bc_lambda:.6f}')

# Save lambda for test-time inverse transform
np.save('boxcox_lambda.npy', bc_lambda)

# ---- Log-transform skewed numerical features ----
log_features = ['GrLivArea', 'TotalBsmtSF', 'LotArea', 'TotalSF']
for col in log_features:
    if col in df.columns:
        df[col] = np.log1p(df[col])

# ---- 10-fold Target Encoding ----
TARGET_ENCODE_COLS = ['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SaleType', 'MSZoning']
SEED = 42
target_encode_means = {}

for col in TARGET_ENCODE_COLS:
    if col not in df.columns:
        continue
    encoded_col_name = f'{col}_TE'
    df[encoded_col_name] = np.nan

    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    for train_idx, val_idx in kf.split(df):
        fold_mean = pd.Series(y[train_idx]).groupby(df[col].iloc[train_idx].values).mean()
        df.loc[val_idx, encoded_col_name] = df.loc[val_idx, col].map(fold_mean)

    global_mean = y.mean()
    df[encoded_col_name] = df[encoded_col_name].fillna(global_mean)

    # Store full-dataset means for test-time
    target_encode_means[col] = pd.Series(y).groupby(df[col]).mean().to_dict()

# Remove target before preprocessing features
X = df.drop(columns=['SalePrice'])

# Identify numeric and categorical features
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['string', 'category']).columns.tolist()

# Standardize numeric features
scaler = StandardScaler()
X_num = pd.DataFrame(scaler.fit_transform(X[num_cols]), columns=num_cols, index=X.index)

# One-hot encode categorical features
if len(cat_cols) > 0:
    X_cat = pd.get_dummies(X[cat_cols], drop_first=False)
    X_processed = pd.concat([X_num, X_cat], axis=1)
else:
    X_processed = X_num

# Save scaler and train columns for consistent test-time preprocessing
joblib.dump(scaler, 'scaler.pkl')
np.save('train_columns.npy', X_processed.columns.tolist())

# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Save results
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)

print(f'\nX_train shape: {X_train.shape}')
print(f'X_val shape:   {X_val.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_val shape:   {y_val.shape}')

# ---- Target encoding stats ----
print('\nTarget encoding (10-fold) correlation with Box-Cox transformed target:')
te_cols = [f'{c}_TE' for c in TARGET_ENCODE_COLS]
for te_col in te_cols:
    if te_col in X.columns:
        corr = X[te_col].corr(pd.Series(y))
        print(f'  {te_col}: {corr:.4f}')

orig_cat_count = sum(df[c].nunique() for c in TARGET_ENCODE_COLS if c in df.columns)
print(f'\nOriginal one-hot columns for {TARGET_ENCODE_COLS}: {orig_cat_count}')
print(f'Target encoding columns: {len(te_cols)} (reducing {orig_cat_count - len(te_cols)} dims)')

# Save target encode means
np.save('target_encode_means.npy', target_encode_means, allow_pickle=True)
print('\nTarget encoding means saved.')
print(f'Box-Cox lambda saved: {bc_lambda:.6f}')
