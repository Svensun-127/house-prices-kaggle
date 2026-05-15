import pandas as pd
import json
import io
import numpy as np

fn = 'train_cleaned.csv'
df = pd.read_csv(fn)
# 1
nulls = int(df.isnull().sum().sum())
# 2
buf = io.StringIO()
df.info(buf=buf)
info = buf.getvalue()
# 3
sp_desc = df['SalePrice'].describe().to_dict()
for k,v in sp_desc.items():
    if isinstance(v, (np.integer,)):
        sp_desc[k] = int(v)
    elif isinstance(v, (np.floating,)):
        sp_desc[k] = float(v)
# 4 anomalies
anoms = {}
if 'GrLivArea' in df.columns:
    gr0 = df[df['GrLivArea']==0]
    anoms['GrLivArea_zero'] = {'count': int(len(gr0)), 'ids': gr0['Id'].astype(int).tolist()[:20]}
else:
    anoms['GrLivArea_zero'] = {'count':0, 'ids':[]}
if 'BedroomAbvGr' in df.columns:
    br0 = df[df['BedroomAbvGr']==0]
    anoms['BedroomAbvGr_zero'] = {'count': int(len(br0)), 'ids': br0['Id'].astype(int).tolist()[:20]}
else:
    anoms['BedroomAbvGr_zero'] = {'count':0, 'ids':[]}

out = {
    'nulls': nulls,
    'info': info,
    'SalePrice_describe': sp_desc,
    'anomalies': anoms
}
print(json.dumps(out, ensure_ascii=False))
