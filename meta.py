from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import Dataset

dataroot=Path('/mnt/datasets/nipa2020_kb')
def hpsearch_lgb(x_tr, y_tr, x_va, y_va):
    n_HP_points_to_test = 100

    from scipy.stats import randint as sp_randint
    from scipy.stats import uniform as sp_uniform
    param_test ={'num_leaves': sp_randint(6, 50),
             'min_child_samples': sp_randint(100, 500),
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8),
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

    reg = lgb.LGBMRegressor(max_depth=3, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=1000)
    gs = RandomizedSearchCV(
        estimator=reg, param_distributions=param_test,
        n_iter=n_HP_points_to_test,
        scoring='neg_root_mean_squared_error',
        cv=3,
        refit=True,
        random_state=314,
        verbose=True)

    fit_params={"early_stopping_rounds":30,
            "eval_metric" : 'rmse',
            "eval_set" : [(x_va, y_va)],
            'eval_names': ['valid'],
            'verbose': 100}
    gs.fit(x_tr, y_tr, **fit_params)
    print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

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

# svr_43_int is not fit at all .. ignored
tr_df.drop('svr_43_int', axis=1, inplace=True)
te_df.drop('svr_43_int', axis=1, inplace=True)

tr_ds = Dataset(tr_df, label=y, free_raw_data=False)
te_ds = Dataset(te_df, free_raw_data=False)

#hpsearch_lgb(x_tr, y_tr, x_va, y_va)
#params = {
#    'learning_rate': 0.02,
#    'max_depth': 3,
#    'boosting': 'gbdt',
#    'objective': 'regression',
#    'metric': 'rmse',
#    'is_training_metric': True,
#    'feature_fraction': 0.9,
#    'bagging_fraction': 0.7,
#    'lambda_l2': 0.7,
#    'bagging_freq': 5,
#    'seed':42
#}
#params.update(
#{
#    'colsample_bytree': 0.5862725825722349, 
#    'min_child_samples': 139, 
#    'min_child_weight': 100.0, 
#    'num_leaves': 37, 
#    'reg_alpha': 50, 
#    'reg_lambda': 50, 
#    'subsample': 0.9441970858100377
#})
#
#m = lgb.train(params, train_set=tr_ds, verbose_eval=100, num_boost_round=475)
#
#te_pred0 = np.clip(m.predict(te_ds.data), 0, 361)
#
#from sklearn.linear_model import LinearRegression
#reg = LinearRegression().fit(tr_df, y)
#te_pred1 = reg.predict(te_df)
#
#te_pred = te_pred0*0.8 + te_pred1*0.2
#print(type(te_pred))
#
#sub = pd.read_csv(dataroot/'sample_submission.csv')
#sub.ad_periods = te_pred
#sub.to_csv('pred/stacking0.csv', index=False)
#
#
