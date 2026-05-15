import pandas as pd
fn = r'd:\学习\Kaggle Project\House Prices\train_featured.csv'
df = pd.read_csv(fn)
print('shape', df.shape)
print('columns count', len(df.columns))
print(df.columns.tolist()[:40])
print(df.columns.tolist()[40:])
print('has SalePrice?', 'SalePrice' in df.columns)
print('GrLivArea min/max', df['GrLivArea'].min(), df['GrLivArea'].max())
print('MoSold unique sample', sorted(df['MoSold'].unique())[:12])
print('Id sample', df['Id'].head(10).tolist())
