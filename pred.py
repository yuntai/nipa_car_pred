import torch
import pandas as pd
from lightgbm import Dataset
import numpy as np

df = torch.load('prep/vanilla0.pt')
dtest = df[df.split=='test']
ids = dtest.pop('id')
m = torch.load('model_full.pt')
cols = [c for c in dtest.columns if c not in ['fold', 'id', 'split', 'ad_periods']]

x_te = dtest[m.feature_name()]
te_ds = Dataset(x_te, free_raw_data=False)
te_pred = m.predict(te_ds.data)

sub = pd.read_csv('data/sample_submission.csv')
assert np.all(sub.id.values == ids.values)

sub.ad_periods = te_pred
sub.to_csv('pred/nov12.csv', index=False)


