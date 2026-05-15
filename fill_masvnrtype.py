import pandas as pd

df = pd.read_csv('train_cleaned.csv')
if 'MasVnrType' not in df.columns:
    raise KeyError('MasVnrType column not found')

missing_before = int(df['MasVnrType'].isnull().sum())
df['MasVnrType'] = df['MasVnrType'].fillna('None')
missing_after = int(df['MasVnrType'].isnull().sum())

missing_total = int(df.isnull().sum().sum())

output = {
    'missing_before': missing_before,
    'missing_after': missing_after,
    'missing_total_after': missing_total,
}

print(output)
df.to_csv('train_final.csv', index=False)
