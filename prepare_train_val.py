import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

# Load featured data
fn = 'train_featured.csv'
df = pd.read_csv(fn)

# Step 1: drop Id
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Step 3: target log transform
if 'SalePrice' not in df.columns:
    raise KeyError('SalePrice column not found')

y = np.log1p(df['SalePrice'])

# Step 2: log-transform skewed numerical features
log_features = ['GrLivArea', 'TotalBsmtSF', 'LotArea', 'TotalSF']
for col in log_features:
    if col in df.columns:
        df[col] = np.log1p(df[col])
    else:
        raise KeyError(f'{col} column not found')

# ---- Target Encoding: 5-fold for Neighborhood and high-cardinality categoricals ----
TARGET_ENCODE_COLS = ['Neighborhood', 'Exterior1st', 'Exterior2nd']
SEED = 42

# Store the full-dataset means for test-time use
target_encode_means = {}

for col in TARGET_ENCODE_COLS:
    if col not in df.columns:
        continue
    encoded_col_name = f'{col}_TE'
    df[encoded_col_name] = np.nan

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_idx, val_idx in kf.split(df):
        fold_mean = y.iloc[train_idx].groupby(df[col].iloc[train_idx]).mean()
        df.loc[val_idx, encoded_col_name] = df.loc[val_idx, col].map(fold_mean)

    # Fill any remaining NaN (categories not seen in a fold) with global mean
    global_mean = y.mean()
    df[encoded_col_name] = df[encoded_col_name].fillna(global_mean)

    # Store full-dataset means for test-time
    target_encode_means[col] = y.groupby(df[col]).mean().to_dict()

# Remove target before preprocessing features
X = df.drop(columns=['SalePrice'])

# Identify numeric and categorical features
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['string', 'category']).columns.tolist()

# Step 4: standardize numeric features
scaler = StandardScaler()
X_num = pd.DataFrame(scaler.fit_transform(X[num_cols]), columns=num_cols, index=X.index)

# Step 5: one-hot encode categorical features
if len(cat_cols) > 0:
    X_cat = pd.get_dummies(X[cat_cols], drop_first=False)
    X_processed = pd.concat([X_num, X_cat], axis=1)
else:
    X_processed = X_num

# Step 6: split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Save results
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)

y_train.to_numpy().dump('y_train.npy')
y_val.to_numpy().dump('y_val.npy')

print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)
print('y_train shape:', y_train.shape)
print('y_val shape:', y_val.shape)

# ---- Output target encoding stats ----
print('\nTarget encoding correlation with SalePrice:')
te_cols = [f'{c}_TE' for c in TARGET_ENCODE_COLS]
for te_col in te_cols:
    if te_col in X.columns:
        corr = X[te_col].corr(y)
        print(f'  {te_col}: {corr:.4f}')

# Original one-hot dimension equivalent
orig_cat_count = sum(df[c].nunique() for c in TARGET_ENCODE_COLS if c in df.columns)
print(f'\nOriginal one-hot columns for {TARGET_ENCODE_COLS}: {orig_cat_count}')
print(f'Target encoding columns: {len(te_cols)} (reducing {orig_cat_count - len(te_cols)} dims)')

# Save target encode means for test time
np.save('target_encode_means.npy', target_encode_means, allow_pickle=True)
print('\nTarget encoding means saved to target_encode_means.npy')
