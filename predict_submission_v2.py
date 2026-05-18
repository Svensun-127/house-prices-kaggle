"""Final submission v2: 7-model stacking ensemble with all v2 enhancements."""
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.special import inv_boxcox

HIGH_MISSING = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
GARAGE_CAT = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
GARAGE_NUM = ['GarageYrBlt', 'GarageCars', 'GarageArea']
BSMT_CAT = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
BSMT_NUM = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
EX_MAP = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
BSMT_MAP = {'NoBsmt': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
TARGET_ENCODE_COLS = ['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SaleType', 'MSZoning']
LOG_FEATURES = ['GrLivArea', 'TotalBsmtSF', 'LotArea', 'TotalSF']


def drop_high_missing(df):
    return df.drop(columns=[c for c in HIGH_MISSING if c in df.columns])


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
        df['LotFrontage'] = df['LotFrontage'].fillna(train_df['LotFrontage'].median())
    if 'Electrical' in df.columns:
        mode_val = train_df['Electrical'].mode(dropna=True)
        df['Electrical'] = df['Electrical'].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'SBrkr')
    return df


def log_transform(df):
    df = df.copy()
    for col in LOG_FEATURES:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df


def apply_target_encode(df, encode_means):
    df = df.copy()
    for col, mapping in encode_means.items():
        if col in df.columns:
            te_col = f'{col}_TE'
            df[te_col] = df[col].map(mapping)
            mean_val = np.mean(list(mapping.values())) if mapping else 0
            df[te_col] = df[te_col].fillna(mean_val)
    return df


def make_processed_matrix(df, scaler, train_columns):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    X_num = pd.DataFrame(scaler.transform(df[num_cols]), columns=num_cols, index=df.index)
    if cat_cols:
        X_cat = pd.get_dummies(df[cat_cols], drop_first=False)
        X_processed = pd.concat([X_num, X_cat], axis=1)
    else:
        X_processed = X_num
    X_processed = X_processed.reindex(columns=train_columns, fill_value=0)
    return X_processed


if __name__ == '__main__':
    print('=== Loading artifacts ===')
    bc_lambda = float(np.load('boxcox_lambda.npy'))
    scaler_all = joblib.load('scaler_all.pkl')
    train_columns = np.load('train_columns_all.npy', allow_pickle=True).tolist()
    te_means = np.load('target_encode_means.npy', allow_pickle=True).item()
    nbhd_stats = pd.read_csv('nbhd_stats.csv')

    # Load data
    from create_features_v2 import create_features_v2
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    test_ids = test['Id'].copy()
    train_feat = pd.read_csv('train_featured_v2.csv')

    # Preprocess test
    print('Preprocessing test data...')
    test_proc = fill_missing_test(test, train)
    test_proc = create_features_v2(test_proc, nbhd_stats=nbhd_stats)
    test_proc = drop_high_missing(test_proc)
    # Fill any remaining NaN (e.g. MasVnrRatio when MasVnrArea=0, GarageAge, etc.)
    for col in test_proc.columns:
        if test_proc[col].dtype in ['float64', 'int64']:
            test_proc[col] = test_proc[col].fillna(0)
        else:
            test_proc[col] = test_proc[col].fillna('None')
    test_proc = log_transform(test_proc)
    test_proc = test_proc.drop(columns=['Id'])
    test_proc = apply_target_encode(test_proc, te_means)
    X_test = make_processed_matrix(test_proc, scaler_all, train_columns)
    print(f'Test matrix shape: {X_test.shape}')

    # Load models
    print('Loading base models...')
    xgb_final = xgb.Booster()
    xgb_final.load_model('xgb_stacking_v2.json')

    lgb_final = lgb.Booster(model_file='lgb_stacking_v2.txt')

    cb_final = CatBoostRegressor()
    cb_final.load_model('cb_stacking_v2.cbm')

    rf_final = joblib.load('rf_stacking_v2.pkl')
    ridge_final = joblib.load('ridge_stacking_v2.pkl')
    lasso_final = joblib.load('lasso_stacking_v2.pkl')
    enet_final = joblib.load('enet_stacking_v2.pkl')

    meta_model = joblib.load('meta_model_v2.pkl')

    # Predict with each base model
    print('Generating base model predictions...')
    dtest = xgb.DMatrix(X_test, feature_names=train_columns)
    xgb_pred = xgb_final.predict(dtest)
    lgb_pred = lgb_final.predict(X_test)
    cb_pred = cb_final.predict(X_test)
    rf_pred = rf_final.predict(X_test)
    ridge_pred = ridge_final.predict(X_test)
    lasso_pred = lasso_final.predict(X_test)
    enet_pred = enet_final.predict(X_test)

    meta_test = np.column_stack([xgb_pred, lgb_pred, cb_pred, rf_pred,
                                  ridge_pred, lasso_pred, enet_pred])
    ensemble_pred_boxcox = meta_model.predict(meta_test)

    # Inverse Box-Cox
    ensemble_pred_price = inv_boxcox(ensemble_pred_boxcox, bc_lambda)
    ensemble_pred_price = np.maximum(ensemble_pred_price, 1)

    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': ensemble_pred_price})
    submission.to_csv('submission_v2.csv', index=False)

    print(f'\nBox-Cox lambda: {bc_lambda:.6f}')
    print(f'Meta-model coefficients: {np.round(meta_model.coef_, 4)}')
    print(f'Predictions range: {ensemble_pred_price.min():.0f} - {ensemble_pred_price.max():.0f}')
    print(f'Mean prediction: {ensemble_pred_price.mean():.0f}')
    print('submission_v2.csv saved successfully.')
    print(submission.head(5).to_string(index=False))
