"""Final optimized submission: Box-Cox, target encoding, stacking ensemble."""
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.special import inv_boxcox

HIGH_MISSING = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
GARAGE_CAT = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
GARAGE_NUM = ['GarageYrBlt', 'GarageCars', 'GarageArea']
BSMT_CAT = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
BSMT_NUM = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
EX_MAP = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
BSMT_MAP = {'NoBsmt': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
TARGET_ENCODE_COLS = ['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SaleType', 'MSZoning']


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
        df['Electrical'] = df['Electrical'].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'SBrkr')
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
        df['LotFrontage'] = df['LotFrontage'].fillna(train_df['LotFrontage'].median())
    if 'Electrical' in df.columns:
        mode_val = train_df['Electrical'].mode(dropna=True)
        df['Electrical'] = df['Electrical'].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'SBrkr')
    return df


def create_features(df):
    df = df.copy()
    nbhd_qual_mean = df.groupby('Neighborhood')['OverallQual'].transform('mean')
    nbhd_grliv_mean = df.groupby('Neighborhood')['GrLivArea'].transform('mean')

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

    # Interaction features
    df['Qual_X_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
    df['Qual_X_TotalSF'] = df['OverallQual'] * df['TotalSF']
    df['Qual_X_LotArea'] = df['OverallQual'] * df['LotArea']
    df['BsmtQual_X_BsmtFinSF1'] = df['BsmtQual'].map(BSMT_MAP).fillna(0).astype(int) * df['BsmtFinSF1']
    df['YearBuilt_X_Qual'] = df['YearBuilt'] * df['OverallQual']
    df['TotalSF2'] = df['TotalSF'] ** 2
    df['Nbhd_X_Qual'] = nbhd_qual_mean * df['OverallQual']
    df['Nbhd_X_GrLivArea'] = nbhd_grliv_mean * df['GrLivArea']
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
            mean_val = np.mean(list(mapping.values())) if mapping else 0
            df[te_col] = df[te_col].fillna(mean_val)
    return df


def compute_target_encode_means(df, y):
    encode_means = {}
    for col in TARGET_ENCODE_COLS:
        if col in df.columns:
            encode_means[col] = pd.Series(y).groupby(df[col]).mean().to_dict()
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
    print('=== Loading data and meta-model ===')
    bc_lambda = float(np.load('boxcox_lambda.npy'))

    # Load the trained meta-model from stacking
    meta_model = np.load('meta_model.npy', allow_pickle=True).item()

    train_feat = pd.read_csv('train_featured.csv')
    test = pd.read_csv('test.csv')
    test_ids = test['Id'].copy()

    # Preprocess training data
    print('Preprocessing training data...')
    train_feat = drop_high_missing(train_feat)
    train_feat = fill_missing_train(train_feat)
    train_feat = create_features(train_feat)

    # Apply outlier removal BEFORE log_transform (on original-scale values)
    mask_outlier = (
        ((train_feat['GrLivArea'] > 4000) & (train_feat['SalePrice'] < 300000)) |
        (train_feat['LotArea'] > 100000) |
        (train_feat['TotalBsmtSF'] > 3000)
    )
    train_feat = train_feat[~mask_outlier].reset_index(drop=True)
    print(f'Training samples after outlier removal: {len(train_feat)}')

    train_feat = log_transform(train_feat)

    train_feat = train_feat.drop(columns=['Id'])
    from scipy.stats import boxcox
    y = boxcox(train_feat['SalePrice'].values, lmbda=bc_lambda)
    X = train_feat.drop(columns=['SalePrice'])

    # Target encoding
    te_means = compute_target_encode_means(X, y)
    X = apply_target_encode(X, te_means)

    # Load saved scaler and train columns (consistent with training)
    scaler = joblib.load('scaler.pkl')
    train_columns = np.load('train_columns.npy', allow_pickle=True).tolist()

    X_full_proc = make_processed_matrix(X, scaler, train_columns)

    # Retrain final models on the full training data with tuned params
    print('Retraining base models on full data with tuned params...')
    import json
    with open('best_params.json') as f:
        tuned = json.load(f)

    xgb_best = tuned['xgb']
    lgb_best = tuned['lgb']
    cb_best = tuned['cb']

    # XGBoost
    xgb_params = {
        'objective': 'reg:squarederror',
        'learning_rate': xgb_best.get('learning_rate', 0.02),
        'max_depth': int(xgb_best.get('max_depth', 5)),
        'subsample': xgb_best.get('subsample', 0.8),
        'colsample_bytree': xgb_best.get('colsample_bytree', 0.8),
        'reg_alpha': xgb_best.get('reg_alpha', 0.1),
        'reg_lambda': xgb_best.get('reg_lambda', 1.0),
        'min_child_weight': xgb_best.get('min_child_weight', 1),
        'seed': 42,
    }
    dtrain_full = xgb.DMatrix(X_full_proc, label=y, feature_names=train_columns)
    xgb_final = xgb.train(xgb_params, dtrain_full, num_boost_round=1000, verbose_eval=False)

    # LightGBM
    lgb_final = lgb.train(
        {'objective': 'regression', 'metric': 'rmse',
         'learning_rate': lgb_best.get('learning_rate', 0.02),
         'max_depth': int(lgb_best.get('max_depth', 5)),
         'num_leaves': int(lgb_best.get('num_leaves', 31)),
         'subsample': lgb_best.get('subsample', 0.8),
         'colsample_bytree': lgb_best.get('colsample_bytree', 0.8),
         'reg_alpha': lgb_best.get('reg_alpha', 0.1),
         'reg_lambda': lgb_best.get('reg_lambda', 1.0),
         'min_child_samples': int(lgb_best.get('min_child_samples', 20)),
         'seed': 42, 'verbose': -1},
        lgb.Dataset(X_full_proc, label=y), num_boost_round=1000,
    )

    # CatBoost
    cb_final = CatBoostRegressor(
        learning_rate=cb_best.get('learning_rate', 0.02),
        depth=int(cb_best.get('depth', 5)),
        subsample=cb_best.get('subsample', 0.8),
        l2_leaf_reg=cb_best.get('l2_leaf_reg', 1.0),
        random_strength=cb_best.get('random_strength', 1.0),
        iterations=1000, random_seed=42, verbose=False,
    )
    cb_final.fit(X_full_proc, y)

    # Random Forest
    rf_final = RandomForestRegressor(
        n_estimators=500, max_depth=12, min_samples_leaf=4,
        random_state=42, n_jobs=-1,
    )
    rf_final.fit(X_full_proc, y)

    # Preprocess test data
    print('Preprocessing test data...')
    test_proc = drop_high_missing(test)
    test_proc = fill_missing_test(test_proc, train_feat)
    test_proc = create_features(test_proc)
    test_proc = log_transform(test_proc)
    test_proc = test_proc.drop(columns=['Id'])
    test_proc = apply_target_encode(test_proc, te_means)
    X_test = make_processed_matrix(test_proc, scaler, train_columns)

    # ---- Stacking prediction ----
    print('Generating stacking predictions...')
    dtest = xgb.DMatrix(X_test, feature_names=train_columns)
    xgb_test_pred = xgb_final.predict(dtest)
    lgb_test_pred = lgb_final.predict(X_test)
    cb_test_pred = cb_final.predict(X_test)
    rf_test_pred = rf_final.predict(X_test)

    meta_test = np.column_stack([xgb_test_pred, lgb_test_pred, cb_test_pred, rf_test_pred])
    ensemble_pred_boxcox = meta_model.predict(meta_test)

    # Inverse Box-Cox transform
    ensemble_pred_price = inv_boxcox(ensemble_pred_boxcox, bc_lambda)

    # Ensure positive
    ensemble_pred_price = np.maximum(ensemble_pred_price, 1)

    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': ensemble_pred_price})
    submission.to_csv('submission_optimized.csv', index=False)

    print(f'\nBox-Cox lambda: {bc_lambda:.6f}')
    print(f'Meta-model coefficients: XGB={meta_model.coef_[0]:.4f}, LGB={meta_model.coef_[1]:.4f}, CB={meta_model.coef_[2]:.4f}, RF={meta_model.coef_[3]:.4f}')
    print(f'Predictions range: {ensemble_pred_price.min():.0f} - {ensemble_pred_price.max():.0f}')
    print('submission_optimized.csv saved successfully.')
    print(submission.head(5).to_string(index=False))
