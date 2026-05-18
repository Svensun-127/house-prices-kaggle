import pandas as pd

# Load the final cleaned dataset
fn = 'train_final.csv'
df = pd.read_csv(fn)

# Create feature mappings
ex_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
bsmt_map = {'NoBsmt': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

# Precompute neighborhood means for interaction features
nbhd_qual_mean = df.groupby('Neighborhood')['OverallQual'].mean()
nbhd_grliv_mean = df.groupby('Neighborhood')['GrLivArea'].mean()

# Create new features
new_cols = {}
new_cols['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'] + df['GarageArea']
new_cols['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
new_cols['HouseAge'] = df['YrSold'] - df['YearBuilt']
new_cols['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
new_cols['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
new_cols['HasGarage'] = (df['GarageArea'] > 0).astype(int)
new_cols['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
new_cols['OverallScore'] = df['OverallQual'] * df['OverallCond']
new_cols['ExteriorScore'] = df['ExterQual'].map(ex_map).fillna(0).astype(int) + df['ExterCond'].map(ex_map).fillna(0).astype(int)
new_cols['BsmtScore'] = df['BsmtQual'].map(bsmt_map).fillna(0).astype(int)

# ---- Interaction Features ----
new_cols['Qual_X_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
new_cols['Qual_X_TotalSF'] = df['OverallQual'] * new_cols['TotalSF']
new_cols['Qual_X_LotArea'] = df['OverallQual'] * df['LotArea']
new_cols['BsmtQual_X_BsmtFinSF1'] = df['BsmtQual'].map(bsmt_map).fillna(0).astype(int) * df['BsmtFinSF1']

# ---- New Interaction Features ----
new_cols['YearBuilt_X_Qual'] = df['YearBuilt'] * df['OverallQual']
new_cols['TotalSF2'] = new_cols['TotalSF'] ** 2
new_cols['Nbhd_X_Qual'] = df['Neighborhood'].map(nbhd_qual_mean) * df['OverallQual']
new_cols['Nbhd_X_GrLivArea'] = df['Neighborhood'].map(nbhd_grliv_mean) * df['GrLivArea']

for col, series in new_cols.items():
    df[col] = series

# Save featured dataset
out_fn = 'train_featured.csv'
df.to_csv(out_fn, index=False)

# Display summary
print('New features (first 5 rows):')
print(df[list(new_cols.keys())].head(5).to_string(index=False))
print('\nNew features statistics:')
print(df[list(new_cols.keys())].describe().T)
