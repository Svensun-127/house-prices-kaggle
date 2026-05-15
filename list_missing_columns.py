import pandas as pd, json

df = pd.read_csv('train_cleaned.csv')
miss = df.isnull().sum()
miss = miss[miss>0].sort_values(ascending=False)
out = {k:int(v) for k,v in miss.items()}
print(json.dumps(out, ensure_ascii=False))
