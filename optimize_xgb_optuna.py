import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUTLIER_IDS = [524, 1299]
LOG_FEATURES = ['GrLivArea', 'TotalBsmtSF', 'LotArea', 'TotalSF']


def add_features(df):
    df = df.copy()
    df['QualArea'] = df['OverallQual'] * df['TotalSF']
    df['BsmtFinRatio'] = df['BsmtFinSF1'] / (df['TotalBsmtSF'] + 1)
    df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    df['AgeGroup'] = (df['HouseAge'] // 10).astype(int)
    df['MoSold_sin'] = np.sin(2 * np.pi * df['MoSold'] / 12)
    df['MoSold_cos'] = np.cos(2 * np.pi * df['MoSold'] / 12)
    return df


def preprocess(df, scaler=None, fit_scaler=False):
    df = df.copy()
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    if 'SalePrice' in df.columns:
        y = np.log1p(df['SalePrice'])
        df = df.drop(columns=['SalePrice'])
    else:
        y = None
    for col in LOG_FEATURES:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if fit_scaler:
        scaler = StandardScaler().fit(df[num_cols])
    X_num = pd.DataFrame(scaler.transform(df[num_cols]), columns=num_cols, index=df.index)
    X_cat = pd.get_dummies(df[cat_cols], drop_first=False)
    X_processed = pd.concat([X_num, X_cat], axis=1)
    return X_processed, y, scaler


def objective(trial, X_train, y_train, X_val, y_val):
    param = {
        'objective': 'reg:squarederror',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0),
        'seed': 42,
        'verbosity': 0,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_train.columns.tolist())
    bst = xgb.train(
        params=param,
        dtrain=dtrain,
        num_boost_round=500,
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    preds = bst.predict(dval)
    score = np.sqrt(mean_squared_error(y_val, preds))
    return score


if __name__ == '__main__':
    train_df = pd.read_csv('train_featured.csv')
    train_df = train_df[~train_df['Id'].isin(OUTLIER_IDS)].reset_index(drop=True)
    train_df = add_features(train_df)
    X_all, y_all, scaler = preprocess(train_df, fit_scaler=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=42,
    )
    print('Training data shape after outlier removal:', X_train.shape)
    print('Validation data shape:', X_val.shape)

    study = optuna.create_study(direction='minimize')
    func = lambda trial: objective(trial, X_train, y_train, X_val, y_val)
    study.optimize(func, n_trials=30, timeout=1800)

    print('\nBest params:')
    for k, v in study.best_params.items():
        print(f'{k}: {v}')
    print(f'Best validation RMSLE: {study.best_value:.6f}')

    best_param = study.best_params.copy()
    best_param.update({'objective': 'reg:squarederror', 'seed': 42, 'verbosity': 0})
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_train.columns.tolist())
    bst = xgb.train(
        best_param,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    train_pred = bst.predict(dtrain)
    val_pred = bst.predict(dval)
    rmsle_train = np.sqrt(mean_squared_error(y_train, train_pred))
    rmsle_val = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f'RMSLE train: {rmsle_train:.6f}')
    print(f'RMSLE val: {rmsle_val:.6f}')

    feat_imp = pd.Series(bst.get_score(importance_type='gain')).sort_values(ascending=False)
    print('\nTop 10 feature importances:')
    for name, score in feat_imp.head(10).items():
        print(f'{name}: {score:.6f}')
