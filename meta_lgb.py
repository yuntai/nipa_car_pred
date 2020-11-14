from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import Dataset
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf

df = pd.read_csv('prep/prep-43-5.csv')
ys = []
for fold_i in range(5):
    ys.append(df[(df.split=='train')&(df.fold == fold_i)]['ad_periods'])
y = pd.concat(ys, axis=0).reset_index(drop=True)

baseroot = Path("./base")

te_dfs = []
tr_dfs = []
for fn in baseroot.glob("*_tr.csv"):
    tr_dfs.append(pd.read_csv(fn))

tr_df = pd.concat(tr_dfs, axis=1)

for fn in baseroot.glob("*_te.csv"):
    te_dfs.append(pd.read_csv(fn))

te_df = pd.concat(te_dfs, axis=1)

#cols = ['lgb_43_int_120', 'lgb_43_base_150', 'lgb_43_logint_150', 'xgb_43_logint', 'xgb_43_base', 'xgb_43_int']
#tr_df = tr_df[cols]
#te_df = te_df[cols]
# svr_43_int is not fit at all .. ignored
cols = [c for c in tr_df.columns if c.startswith('lgb_43_base_150')]
tr_df = tr_df[cols] 
te_df = te_df[cols] 
