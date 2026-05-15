import pandas as pd
fn = r'd:\学习\Kaggle Project\House Prices\train_featured.csv'
df = pd.read_csv(fn)
print(df.sort_values('GrLivArea', ascending=False)[['Id','GrLivArea','SalePrice']].head(20).to_string(index=False))
print('count > 4000', (df['GrLivArea'] > 4000).sum())
print(df[df['GrLivArea'] > 4000][['Id','GrLivArea','SalePrice']].to_string(index=False))
