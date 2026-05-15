import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

HIGH_MISSING = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
GARAGE_CAT = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
GARAGE_NUM = ['GarageYrBlt', 'GarageCars', 'GarageArea']
BSMT_CAT = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
BSMT_NUM = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
EX_MAP = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
BSMT_MAP = {'NoBsmt': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

TARGET_ENCODE_COLS = ['Neighborhood', 'Exterior1st', 'Exterior2nd']


def drop_high_missing(df):
    return df.drop(columns=[c for c in HIGH_MISSING if c in df.columns])


def fill_missing_train(df):
    df = df.copy()
    for col in GARAGE_CAT:
        if col in df.columns:
            df[col] = df[col].fillna('NoGarage')
    for col in GARAGE_NUM:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in BSMT_CAT:
        if col in df.columns:
            df[col] = df[col].fillna('NoBsmt')
    for col in BSMT_NUM:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if 'MasVnrType' in df.columns:
        df['MasVnrType'] = df['MasVnrType'].fillna('None')
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    if 'Electrical' in df.columns:
        mode_val = df['Electrical'].mode(dropna=True)
        if len(mode_val) > 0:
            mode_val = mode_val.iloc[0]
        else:
            mode_val = 'SBrkr'
        df['Electrical'] = df['Electrical'].fillna(mode_val)
    return df


def fill_missing_test(df, train_df):
    df = df.copy()
    for col in GARAGE_CAT:
        if col in df.columns:
            df[col] = df[col].fillna('NoGarage')
    for col in GARAGE_NUM:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in BSMT_CAT:
        if col in df.columns:
            df[col] = df[col].fillna('NoBsmt')
    for col in BSMT_NUM:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if 'MasVnrType' in df.columns:
        df['MasVnrType'] = df['MasVnrType'].fillna('None')
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    if 'LotFrontage' in df.columns:
        median_by_nbhd = train_df.groupby('Neighborhood')['LotFrontage'].median()
        df['LotFrontage'] = df.apply(
            lambda row: median_by_nbhd.get(row['Neighborhood'], np.nan)
            if pd.isna(row['LotFrontage']) else row['LotFrontage'],
            axis=1,
        )
        overall = train_df['LotFrontage'].median()
        df['LotFrontage'] = df['LotFrontage'].fillna(overall)
    if 'Electrical' in df.columns:
        mode_val = train_df['Electrical'].mode(dropna=True)
        if len(mode_val) > 0:
            mode_val = mode_val.iloc[0]
        else:
            mode_val = 'SBrkr'
        df['Electrical'] = df['Electrical'].fillna(mode_val)
    return df


def create_features(df):
    df = df.copy()
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'] + df['GarageArea']
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['OverallScore'] = df['OverallQual'] * df['OverallCond']
    df['ExteriorScore'] = df['ExterQual'].map(EX_MAP).fillna(0).astype(int) + df['ExterCond'].map(EX_MAP).fillna(0).astype(int)
    df['BsmtScore'] = df['BsmtQual'].map(BSMT_MAP).fillna(0).astype(int)
    # ---- Interaction Features ----
    df['Qual_X_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
    df['Qual_X_TotalSF'] = df['OverallQual'] * df['TotalSF']
    df['Qual_X_LotArea'] = df['OverallQual'] * df['LotArea']
    df['BsmtQual_X_BsmtFinSF1'] = df['BsmtQual'].map(BSMT_MAP).fillna(0).astype(int) * df['BsmtFinSF1']
    return df


def log_transform(df):
    df = df.copy()
    for col in ['GrLivArea', 'TotalBsmtSF', 'LotArea', 'TotalSF']:
        df[col] = np.log1p(df[col])
    return df


def apply_target_encode(df, encode_means):
    df = df.copy()
    for col, mapping in encode_means.items():
        if col in df.columns:
            te_col = f'{col}_TE'
            df[te_col] = df[col].map(mapping)
            df[te_col] = df[te_col].fillna(np.mean(list(mapping.values())) if mapping else 0)
    return df


def compute_target_encode_means(df, y):
    encode_means = {}
    for col in TARGET_ENCODE_COLS:
        if col in df.columns:
            encode_means[col] = y.groupby(df[col]).mean().to_dict()
    return encode_means


def make_processed_matrix(df, scaler, train_columns):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    X_num = pd.DataFrame(scaler.transform(df[num_cols]), columns=num_cols, index=df.index)
    X_cat = pd.get_dummies(df[cat_cols], drop_first=False)
    X_processed = pd.concat([X_num, X_cat], axis=1)
    X_processed = X_processed.reindex(columns=train_columns, fill_value=0)
    return X_processed


if __name__ == '__main__':
    # Load ensemble weights
    weights = np.load('ensemble_weights.npy')

    # Load models
    xgb_model = xgb.Booster()
    xgb_model.load_model('xgb_model.json')
    lgb_model = lgb.Booster(model_file='lgb_model.txt')
    cb_model = CatBoostRegressor()
    cb_model.load_model('cb_model.cbm')

    # Use training columns from X_train.csv as the reference
    X_train_ref = pd.read_csv('X_train.csv')
    train_columns = X_train_ref.columns.tolist()

    train_feat = pd.read_csv('train_featured.csv')
    test = pd.read_csv('test.csv')
    test_ids = test['Id'].copy()

    # Preprocess train data to derive scaler
    train_feat = drop_high_missing(train_feat)
    train_feat = fill_missing_train(train_feat)
    train_feat = create_features(train_feat)
    train_feat = log_transform(train_feat)
    train_feat = train_feat.drop(columns=['Id'])
    y_full = np.log1p(train_feat['SalePrice'])
    X_full = train_feat.drop(columns=['SalePrice'])

    # Target encoding (compute means from full training set)
    te_means = compute_target_encode_means(X_full, y_full)
    X_full = apply_target_encode(X_full, te_means)

    num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_full.select_dtypes(exclude=[np.number]).columns.tolist()
    scaler = StandardScaler().fit(X_full[num_cols])
    X_num = pd.DataFrame(scaler.transform(X_full[num_cols]), columns=num_cols, index=X_full.index)
    X_cat = pd.get_dummies(X_full[cat_cols], drop_first=False)
    X_full_proc = pd.concat([X_num, X_cat], axis=1)
    X_full_proc = X_full_proc.reindex(columns=train_columns, fill_value=0)

    # Fit models on full training data for best submission
    # Use best iteration counts from validation (safe estimates)
    XGB_BEST_ROUNDS = 300
    LGB_BEST_ROUNDS = 410
    CB_BEST_ROUNDS = 500

    dtrain_full = xgb.DMatrix(X_full_proc, label=y_full, feature_names=train_columns)
    xgb_params = {
        'objective': 'reg:squarederror', 'learning_rate': 0.03, 'max_depth': 4,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'seed': 42,
    }
    xgb_bst = xgb.train(xgb_params, dtrain_full, num_boost_round=XGB_BEST_ROUNDS, verbose_eval=False)

    lgb_train_full = lgb.Dataset(X_full_proc, label=y_full)
    lgb_bst = lgb.train(
        {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.03, 'max_depth': 4,
         'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8,
         'reg_alpha': 0.1, 'reg_lambda': 1.0, 'seed': 42, 'verbose': -1},
        lgb_train_full, num_boost_round=LGB_BEST_ROUNDS,
    )

    cb_bst = CatBoostRegressor(
        learning_rate=0.03, depth=4, iterations=CB_BEST_ROUNDS,
        subsample=0.8, random_seed=42, verbose=False,
    )
    cb_bst.fit(X_full_proc, y_full)

    # Preprocess test data
    test_proc = drop_high_missing(test)
    test_proc = fill_missing_test(test_proc, train_feat)
    test_proc = create_features(test_proc)
    test_proc = log_transform(test_proc)
    test_proc = test_proc.drop(columns=['Id'])
    test_proc = apply_target_encode(test_proc, te_means)
    X_test = make_processed_matrix(test_proc, scaler, train_columns)

    # ---- Ensemble prediction ----
    dtest = xgb.DMatrix(X_test, feature_names=train_columns)
    xgb_pred = xgb_bst.predict(dtest)
    lgb_pred = lgb_bst.predict(X_test)
    cb_pred = cb_bst.predict(X_test)

    ensemble_pred_log = weights[0] * xgb_pred + weights[1] * lgb_pred + weights[2] * cb_pred
    ensemble_pred_price = np.expm1(ensemble_pred_log)

    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': ensemble_pred_price})
    submission.to_csv('submission_ensemble.csv', index=False)

    print(f'Ensemble weights: XGB={weights[0]:.3f}, LGB={weights[1]:.3f}, CB={weights[2]:.3f}')
    print('submission_ensemble.csv saved successfully.')
    print(submission.head(5).to_string(index=False))
