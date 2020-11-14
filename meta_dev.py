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


def fit_dnn(x_tr, y_tr):
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(x_tr))

    def build_and_compile_model(norm):
        model = keras.Sequential([
            norm,
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32),
            layers.Dense(1)
        ])

        model.compile(loss='mean_squared_error',
                    optimizer=tf.keras.optimizers.Adam(0.0005))
        return model

    m = build_and_compile_model(normalizer)

    history = m.fit(
        x_tr, y_tr,
        epochs=30,
        # suppress logging
        verbose=1,
        # Calculate validation results on 20% of the training data
        validation_split = 0.1
    )

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
te_df = te_df[tr_df.columns]

# svr_43_int is not fit at all .. ignored
tr_df.drop('svr_43_int', axis=1, inplace=True)
te_df.drop('svr_43_int', axis=1, inplace=True)

#cols = ['lgb_43_int_120', 'lgb_43_base_150', 'lgb_43_logint_150', 'xgb_43_logint', 'xgb_43_base', 'xgb_43_int']
#tr_df = tr_df[cols]
#te_df = te_df[cols]
# svr_43_int is not fit at all .. ignored
#cols = [c for c in tr_df.columns if c.startswith('lgb')]
#tr_df = tr_df[cols] 
#te_df = te_df[cols] 
#tr_df.drop(cols_to_drop, axis=1, inplace=True)
#te_df.drop(cols_to_drop, axis=1, inplace=True)

x_tr, x_va, y_tr, y_va = train_test_split(tr_df, y, test_size=0.1, shuffle=True, random_state=42)

tr_ds = Dataset(x_tr, label=y_tr, free_raw_data=False)
va_ds = Dataset(x_va, label=y_va, free_raw_data=False)

valid_sets = [tr_ds, va_ds]

#hpsearch_lgb(x_tr, y_tr, x_va, y_va)
params = {
    'learning_rate': 0.001,
    'max_depth': -1,
    'boosting': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'is_training_metric': True,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'lambda_l2': 0.2,
    'bagging_freq': 5,
    'seed':42
}

params.update(
{
    'colsample_bytree': 0.5862725825722349, 
    'min_child_samples': 139, 
    'min_child_weight': 100.0, 
    'num_leaves': 37, 
    'reg_alpha': 50, 
    'reg_lambda': 50, 
    'subsample': 0.9441970858100377
})

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.005
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

m = lgb.train(params, early_stopping_rounds=1000, valid_sets=valid_sets, 
              train_set=tr_ds, verbose_eval=100, num_boost_round=10000,
              callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)]
             )
va_pred = m.predict(x_va)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_tr, y_tr)
lin_pred = reg.predict(x_va)

for i in range(11):
    f = i/10.
    pred = va_pred * f + lin_pred*(1-f)
    pred = np.clip(pred, 0, 361)

    va_score = np.sqrt(mean_squared_error(pred, y_va))
    print(f, va_score)



