import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler

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

    reg = lgb.LGBMRegressor(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)
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
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100}
            #'categorical_feature': 'auto'}
    gs.fit(x_tr, y_tr, **fit_params)
    print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.02
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def fit_lgb(x_tr, y_tr, x_va, y_va, cat_feats, args):
    from lightgbm import Dataset

    if args.clip_target != -1:
        y_tr = y_tr.clip(upper=args.clip_target)

    tr_ds = Dataset(x_tr, label=y_tr, free_raw_data=False)
    if args.mode not in ['full', 'fold']:
        va_ds = Dataset(x_va, label=y_va, free_raw_data=False)
        valid_sets = [tr_ds, va_ds]
    else:
        valid_sets = [tr_ds]

    params = {
        'learning_rate': 0.02,
        'max_depth': -1,
        'boosting': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'is_training_metric': True,
        'num_leaves': args.num_leaves,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'lambda_l2': 0.7,
        'bagging_freq': 5,
        'seed':42
    }

    kwargs = {
        'train_set': tr_ds,
        'categorical_feature': cat_feats,
        'verbose_eval': args.verbose_eval,
        'num_boost_round': args.num_boost_round,
    }

    if args.mode not in ['full', 'fold']:
        kwargs['early_stopping_rounds'] = 200
        kwargs['valid_sets'] = valid_sets

    if args.lr_decay:
        kwargs['callbacks'] = [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)]

    m = lgb.train(params, **kwargs)

    tr_pred = np.clip(m.predict(tr_ds.data), 0, 361)
    tr_score  = np.sqrt(mean_squared_error(tr_pred, tr_ds.label))

    if args.mode not in ['full', 'fold']:
        va_pred = np.clip(m.predict(va_ds.data), 0, 361)
        va_score = np.sqrt(mean_squared_error(va_pred, va_ds.label))
    else:
        va_score = 0.

    return m, tr_score, va_score

def fit_xgb(x_tr, y_tr, x_va, y_va, cat_feats, args):
    import xgboost as xgb

    dtrain = xgb.DMatrix(data=x_tr, label=y_tr)
    dvalid = xgb.DMatrix(data=x_va, label=y_va)

    params = {
        'objective': 'reg:squarederror',
        'eta': 0.01,
        'max_depth':16,
        'eval_metric': ["rmse"],
        'nthread':16,
    }
    if args.mode not in ['full', 'fold']:
        nround = 5000
        eval_list = [(dtrain, 'train'), (dvalid, 'val')]
        nround = args.num_boost_round
        m = xgb.train(params, dtrain, 
                           nround, 
                           evals=eval_list, 
                           verbose_eval=100,
                           early_stopping_rounds=200)
        va_pred = np.clip(m.predict(dvalid), 0, 361)
        va_score = np.sqrt(mean_squared_error(y_va, va_pred))
    else:
        nround = args.num_boost_round
        m = xgb.train(params, dtrain, nround, verbose_eval=100)
        va_score = 0

    tr_pred = np.clip(m.predict(dtrain), 0, 361)
    tr_score = np.sqrt(mean_squared_error(y_tr, tr_pred))

    return m, tr_score, va_score

def crossvaltest():
    params = {'depth':[3,1,2,6,4,5,7,8,9,10],
          'iterations':[250,100,500,1000],
          'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3],
          'l2_leaf_reg':[3,1,5,10,100],
          'border_count':[32,5,10,20,50,100,200],
          'ctr_border_count':[50,5,10,20,100,200],
          'thread_count':4}

def fit_cb(x_tr, y_tr, x_va, y_va, cat_feats, args):
    import catboost as cb

    params = {
        'iterations': args.num_boost_round,
        'task_type': 'GPU',
        #'devices': '0:1',
        'loss_function': 'RMSE',
        'max_depth': 8,
        'cat_features': cat_feats,
    }

    m = cb.CatBoostRegressor(**params)
    if args.mode not in ['full', 'fold']:
        m.fit(x_tr, y_tr, eval_set=(x_va, y_va), verbose=100)
        va_pred = np.clip(m.predict(x_va), 0, 361)
        val_score = np.sqrt(mean_squared_error(y_va, va_pred))
    else:
        m.fit(x_tr, y_tr, verbose=1000)
        val_score = 0.

    tr_pred = np.clip(m.predict(x_tr), 0, 361)
    tr_score = np.sqrt(mean_squared_error(y_tr, tr_pred))

    m.best_iteration = m.get_best_iteration()

    return m, tr_score, val_score

def hpsearch_rf(x_tr, y_tr, x_va, y_va, cat_feats, args):
    from sklearn.model_selection import RandomizedSearchCV
    from pprint import pprint
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    pprint(random_grid)
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(x_tr, y_tr)
    print(rf_random.best_params_)

def fit_rf(x_tr, y_tr, x_va, y_va, cat_feats, args):

    params = {
        "n_estimators":300,
        "n_jobs": 8,
        "random_state":5436,
        "verbose": 1
    }

    rf = RandomForestRegressor(**params)
    rf.fit(x_tr, y_tr)

    va_score = 0.
    if args.mode not in ['full', 'fold']:
        y_pred = rf.predict(x_va)
        va_score = np.sqrt(mean_squared_error(y_pred, y_va))

    y_pred = rf.predict(x_tr)
    tr_score = np.sqrt(mean_squared_error(y_pred, y_tr))

    feature_names = x_tr.columns.tolist()
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names),  reverse=True)[:20])
    rf.best_iteration = None

    return rf, tr_score, va_score

def hpsearch_svr(x_tr, y_tr, x_va, y_va, cat_feats, args):
    from sklearn.svm import SVR
    import numpy as np
    from sklearn.pipeline import Pipeline
    scaler = StandardScaler()
    svr = LinearSVR(random_state=0, tol=1e-5)

    parameters = {'svr__C':[1.5, 10],'svr__epsilon':[0.1,0.2,0.5,0.3]}
    regr = Pipeline(steps=[('scaler', scaler), ('svr', svr)])
    regr = GridSearchCV(regr, parameters)
    regr.fit(x_tr, y_tr)
    print(regr.best_params_)

def fit_svr(x_tr, y_tr, x_va, y_va, cat_feats, args):
    regr = make_pipeline(StandardScaler(),
            LinearSVR(random_state=0, tol=1e-5, C=10, epsilon=0.3))
    regr.fit(x_tr, y_tr)

    if args.mode not in ['full', 'fold']:
        y_pred = regr.predict(x_va)
        va_score = np.sqrt(mean_squared_error(y_pred, y_va))

    y_pred = regr.predict(x_tr)
    tr_score = np.sqrt(mean_squared_error(y_pred, y_tr))
    va_score = 0.

    regr.best_iteration = None

    return regr, tr_score, va_score

# not used
def fit_tabnet(x_tr, y_tr, x_va, y_va, cat_feats, args):
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

    cat_idxs = [x_tr.columns.get_loc(f) for f in cat_feats]
    cat_dims = x_tr[cat_feats].apply(lambda s: s.nunique()).tolist()

    cat_emb_dim = [i//2 for i in cat_dims]

    x_tr = x_tr.values
    y_tr = y_tr.values.reshape(-1, 1)

    x_va = x_va.values
    y_va = y_va.values.reshape(-1, 1)
    
    params = dict(n_d=16, n_a=16, n_steps=3,
        gamma=1.5, n_independent=4, n_shared=4,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims, 
        cat_emb_dim=cat_emb_dim,
        lambda_sparse=0.0001, momentum=0.95, clip_value=2.,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.0005),
        #scheduler_params = {"gamma": 0.95, "step_size": 500},
        scheduler_params = {"gamma": 0.95},
        scheduler_fn=torch.optim.lr_scheduler.ExponentialLR,
                  epsilon=1e-1)

    clf = TabNetRegressor(**params)
    fit_params = {
        'batch_size': 4096, 
        'virtual_batch_size': 1024,
        'eval_set':[(x_va, y_va)],
        'max_epochs': 1000,
        'patience':50,
    }

    clf.fit(
      x_tr, y_tr,
      **fit_params 
    )


    tr_pred = np.clip(clf.predict(x_tr), 0, 361)
    va_pred = np.clip(clf.predict(x_va), 0, 361)
    train_score = np.sqrt(mean_squared_error(tr_pred, y_tr))
    val_score = np.sqrt(mean_squared_error(va_pred, y_va))

    return clf, train_score, val_score

# not used
def fit_dnn(x_tr, y_tr):
    print(x_tr.columns)
    x_tr = x_tr[['va08', 'va09', 'va10', 'va14', 'va15', 'va16', 'va17', 'va18', 'va19', 'va20', 'va21', 'va29', 'va30', 'va31', 'va34']]
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(x_tr))

    def build_and_compile_model(norm):
        model = keras.Sequential([
            norm,
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mean_squared_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    m = build_and_compile_model(normalizer)

    history = m.fit(
        x_tr, y_tr,
        epochs=30,
        # suppress logging
        verbose=1,
        # Calculate validation results on 20% of the training data
        validation_split = 0.2
    )
