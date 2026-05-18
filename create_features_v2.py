"""Enhanced feature engineering v2: ratio, aggregation, ordinal, and temporal features.
Reads train.csv, outputs train_featured_v2.csv.
Functions are importable for test-time use.
"""
import numpy as np
import pandas as pd

EX_MAP = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
BSMT_MAP = {'NoBsmt': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
QUALITY_MAP = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
FUNCTIONAL_MAP = {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}
PAVED_MAP = {'Y': 3, 'P': 2, 'N': 1}


def create_features_v2(df, nbhd_stats=None):
    """Create all v2 features.
    nbhd_stats: pre-computed neighborhood aggregation DataFrame from training data.
                If None, aggregations are computed from df itself (for training mode)."""
    df = df.copy()

    # ---- Base features (from v1) ----
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

    # ==== NEW v2 Features ====

    # -- Ratio features --
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    df['LivAreaRatio'] = df['GrLivArea'] / (df['LotArea'] + 1)
    df['BsmtFinRatio'] = (df['BsmtFinSF1'] + df['BsmtFinSF2']) / (df['TotalBsmtSF'] + 1)
    df['GaragePerCar'] = df['GarageArea'] / (df['GarageCars'].replace(0, np.nan)).fillna(1)
    df['PorchRatio'] = df['TotalPorchSF'] / (df['LotArea'] + 1)
    df['BathPerRoom'] = df['TotalBath'] / (df['TotRmsAbvGrd'] + 1)
    df['BedroomRatio'] = df['BedroomAbvGr'] / (df['TotRmsAbvGrd'] + 1)
    df['RoomPerSqFt'] = df['TotRmsAbvGrd'] / (df['GrLivArea'] + 1)
    df['QualPerSqFt'] = df['OverallQual'] / (df['GrLivArea'] + 1)
    df['RemodRatio'] = (df['YearRemodAdd'] - df['YearBuilt']) / (df['HouseAge'] + 1)
    df['LivAreaPerRoom'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)
    df['MasVnrRatio'] = df['MasVnrArea'] / (df['GrLivArea'] + 1)
    df['WoodDeckRatio'] = df['WoodDeckSF'] / (df['LotArea'] + 1)
    df['YearsSinceRemod'] = df['YrSold'] - df['YearRemodAdd']

    # GarageAge: fill missing GarageYrBlt with YearBuilt
    garage_yr = df['GarageYrBlt'].copy()
    garage_yr = garage_yr.fillna(df['YearBuilt'])
    df['GarageAge'] = df['YrSold'] - garage_yr

    # -- Ordinal quality features --
    def safe_map(series, mapping, default=0):
        return series.map(mapping).fillna(default).astype(int)

    df['KitchenQual_Ord'] = safe_map(df['KitchenQual'], QUALITY_MAP)
    df['HeatingQC_Ord'] = safe_map(df['HeatingQC'], QUALITY_MAP)
    df['GarageQual_Ord'] = safe_map(df['GarageQual'], QUALITY_MAP)
    df['GarageCond_Ord'] = safe_map(df['GarageCond'], QUALITY_MAP)
    df['BsmtCond_Ord'] = safe_map(df['BsmtCond'], QUALITY_MAP)
    df['FireplaceQu_Ord'] = safe_map(df['FireplaceQu'], QUALITY_MAP)
    df['Functional_Ord'] = safe_map(df['Functional'], FUNCTIONAL_MAP, default=4)
    df['PavedDrive_Ord'] = safe_map(df['PavedDrive'], PAVED_MAP)
    df['BsmtExposure_Ord'] = df['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}).fillna(0).astype(int)
    df['BsmtFinType1_Ord'] = df['BsmtFinType1'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}).fillna(0).astype(int)
    df['BsmtFinType2_Ord'] = df['BsmtFinType2'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}).fillna(0).astype(int)
    df['LotShape_Ord'] = df['LotShape'].map({'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1}).fillna(0).astype(int)
    df['LandSlope_Ord'] = df['LandSlope'].map({'Gtl': 3, 'Mod': 2, 'Sev': 1}).fillna(0).astype(int)

    # Total quality score
    quality_cols = ['OverallQual', 'OverallCond', 'ExterQual_Ord', 'ExterCond_Ord',
                    'KitchenQual_Ord', 'HeatingQC_Ord', 'GarageQual_Ord', 'GarageCond_Ord',
                    'BsmtQual_Ord', 'BsmtCond_Ord', 'FireplaceQu_Ord']
    df['ExterQual_Ord'] = safe_map(df['ExterQual'], QUALITY_MAP)
    df['ExterCond_Ord'] = safe_map(df['ExterCond'], QUALITY_MAP)
    df['BsmtQual_Ord'] = safe_map(df['BsmtQual'], QUALITY_MAP)
    df['TotalQualScore'] = df[quality_cols].sum(axis=1)

    # -- Neighborhood aggregation features --
    if nbhd_stats is not None:
        df = df.merge(nbhd_stats, on='Neighborhood', how='left')
    else:
        nbhd_agg = df.groupby('Neighborhood').agg(
            Nbhd_GrLivArea_mean=('GrLivArea', 'mean'),
            Nbhd_GrLivArea_std=('GrLivArea', 'std'),
            Nbhd_TotalSF_mean=('TotalSF', 'mean'),
            Nbhd_TotalSF_std=('TotalSF', 'std'),
            Nbhd_LotArea_mean=('LotArea', 'mean'),
            Nbhd_LotArea_std=('LotArea', 'std'),
            Nbhd_OverallQual_mean=('OverallQual', 'mean'),
            Nbhd_OverallQual_std=('OverallQual', 'std'),
            Nbhd_YearBuilt_mean=('YearBuilt', 'mean'),
            Nbhd_YearBuilt_std=('YearBuilt', 'std'),
            Nbhd_Count=('GrLivArea', 'count'),
        ).reset_index()
        df = df.merge(nbhd_agg, on='Neighborhood', how='left')

    # Z-scores
    df['GrLivArea_Z'] = (df['GrLivArea'] - df['Nbhd_GrLivArea_mean']) / (df['Nbhd_GrLivArea_std'] + 1)
    df['TotalSF_Z'] = (df['TotalSF'] - df['Nbhd_TotalSF_mean']) / (df['Nbhd_TotalSF_std'] + 1)
    df['LotArea_Z'] = (df['LotArea'] - df['Nbhd_LotArea_mean']) / (df['Nbhd_LotArea_std'] + 1)
    df['OverallQual_Z'] = (df['OverallQual'] - df['Nbhd_OverallQual_mean']) / (df['Nbhd_OverallQual_std'] + 1)
    df['YearBuilt_Z'] = (df['YearBuilt'] - df['Nbhd_YearBuilt_mean']) / (df['Nbhd_YearBuilt_std'] + 1)

    # -- Temporal features --
    df['MoSold_sin'] = np.sin(2 * np.pi * df['MoSold'] / 12)
    df['MoSold_cos'] = np.cos(2 * np.pi * df['MoSold'] / 12)
    df['DecadeBuilt'] = (df['YearBuilt'] // 10) * 10
    df['DecadeRemod'] = (df['YearRemodAdd'] // 10) * 10
    df['YrSold_minus_YearBuilt'] = df['YrSold'] - df['YearBuilt']

    # -- Additional interactions --
    df['OverallQual_X_TotalBath'] = df['OverallQual'] * df['TotalBath']
    df['GrLivArea_X_TotalBath'] = df['GrLivArea'] * df['TotalBath']
    df['OverallQual_X_Fireplaces'] = df['OverallQual'] * df['Fireplaces']
    df['GarageCars_X_GarageArea'] = df['GarageCars'] * df['GarageArea']

    # Replace inf/-inf from division by zero (e.g. RemodRatio when HouseAge=-1)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], 0)

    return df


if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Process train; extract nbhd_stats for reuse on test
    train_out = create_features_v2(train)
    agg_cols = ['Nbhd_GrLivArea_mean', 'Nbhd_GrLivArea_std',
                'Nbhd_TotalSF_mean', 'Nbhd_TotalSF_std',
                'Nbhd_LotArea_mean', 'Nbhd_LotArea_std',
                'Nbhd_OverallQual_mean', 'Nbhd_OverallQual_std',
                'Nbhd_YearBuilt_mean', 'Nbhd_YearBuilt_std',
                'Nbhd_Count']
    nbhd_stats = train_out[['Neighborhood'] + agg_cols].drop_duplicates('Neighborhood').reset_index(drop=True)
    nbhd_stats.to_csv('nbhd_stats.csv', index=False)

    # Process test with train's aggregation stats
    test_out = create_features_v2(test, nbhd_stats=nbhd_stats)

    # Save
    train_out.to_csv('train_featured_v2.csv', index=False)
    test_out.to_csv('test_featured_v2.csv', index=False)

    # Stats
    new_v2_features = [
        'TotalPorchSF', 'LivAreaRatio', 'BsmtFinRatio', 'GaragePerCar', 'PorchRatio',
        'BathPerRoom', 'BedroomRatio', 'RoomPerSqFt', 'QualPerSqFt', 'RemodRatio',
        'LivAreaPerRoom', 'MasVnrRatio', 'WoodDeckRatio', 'YearsSinceRemod', 'GarageAge',
        'KitchenQual_Ord', 'HeatingQC_Ord', 'GarageQual_Ord', 'GarageCond_Ord',
        'BsmtCond_Ord', 'FireplaceQu_Ord', 'Functional_Ord', 'PavedDrive_Ord',
        'BsmtExposure_Ord', 'BsmtFinType1_Ord', 'BsmtFinType2_Ord', 'LotShape_Ord',
        'LandSlope_Ord', 'ExterQual_Ord', 'ExterCond_Ord', 'BsmtQual_Ord', 'TotalQualScore',
        'Nbhd_GrLivArea_mean', 'Nbhd_GrLivArea_std', 'Nbhd_TotalSF_mean', 'Nbhd_TotalSF_std',
        'Nbhd_LotArea_mean', 'Nbhd_LotArea_std', 'Nbhd_OverallQual_mean', 'Nbhd_OverallQual_std',
        'Nbhd_YearBuilt_mean', 'Nbhd_YearBuilt_std', 'Nbhd_Count',
        'GrLivArea_Z', 'TotalSF_Z', 'LotArea_Z', 'OverallQual_Z', 'YearBuilt_Z',
        'MoSold_sin', 'MoSold_cos', 'DecadeBuilt', 'DecadeRemod', 'YrSold_minus_YearBuilt',
        'OverallQual_X_TotalBath', 'GrLivArea_X_TotalBath', 'OverallQual_X_Fireplaces',
        'GarageCars_X_GarageArea',
    ]
    print(f'Added {len(new_v2_features)} new features')
    print(f'Train shape: {train_out.shape}')
    print(f'Test shape:  {test_out.shape}')

    # Check for infinite values
    num_cols = train_out.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(train_out[num_cols]).sum().sum()
    if inf_count > 0:
        print(f'WARNING: {inf_count} infinite values in train')
        inf_cols = train_out[num_cols].columns[np.isinf(train_out[num_cols]).any()].tolist()
        print(f'  Columns with inf: {inf_cols}')
    else:
        print('No infinite values detected.')
