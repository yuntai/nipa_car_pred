import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from tqdm import tqdm
from numpy.random import normal
from argparse import ArgumentParser

import torch
import itertools
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold

def extract_col_interaction(df, col1, col2, tfidf=False):
    # from https://www.kaggle.com/dmitrylarko/kaggledays-sf-2-amazon-unsupervised-encoding
    data = df.groupby([col1])[col2].agg(lambda x: " ".join(list([str(y) for y in x])))
    if tfidf:
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(" "))
    else:
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(" "))

    data_X = vectorizer.fit_transform(data)
    dim_red = TruncatedSVD(n_components=1, random_state = 5115)
    data_X = dim_red.fit_transform(data_X)

    result = pd.DataFrame()
    result[col1] = data.index.values
    if tfidf:
        result[col1+"_{}_tfidf_svd".format(col2)] = data_X.ravel()
    else:
        result[col1+"_{}_svd".format(col2)] = data_X.ravel()
    return result


def get_interactions_svd(df, cols, tfidf=False):
    # from https://www.kaggle.com/dmitrylarko/kaggledays-sf-2-amazon-unsupervised-encoding
    new_dataset = pd.DataFrame()
    for col1, col2 in tqdm(itertools.permutations(cols, 2)):
        data = extract_col_interaction(df, col1, col2, tfidf)
        col_name = [x for x in data.columns if "svd" in x][0]
        new_dataset[col_name] = df[[col1]].merge(data, on=col1, how='left')[col_name]
    return new_dataset

def get_freq_encoding(df, cols):
    new_df = pd.DataFrame()
    for c in cols:
        data = df.groupby([c]).size().reset_index()
        new_df[c+"_freq"] = df[[c]].merge(data, on=c, how="left")[0]
    return new_df

def do_split_(df, fold_label, test_size=0.1):
    ix = int(df.shape[0]*test_size)
    df['fold'] = -1
    df.iloc[ix:,  df.columns.get_loc('fold')] = 0

def do_folding_(df, fold_label, rs, n_splits=5):
    kf = KFold(n_splits=n_splits, random_state=rs, shuffle=True)
    for fold_ix, (_, va_ix) in enumerate(kf.split(df)):
        df.loc[va_ix, fold_label] = fold_ix

def add_interaction_(df, cols):
    for c1, c2 in itertools.combinations(cols, 2):
        df[f'i{c1}d{c2}'] = df[c1]/(df[c2] + 0.0001)
        df[f'i{c2}d{c1}'] = df[c2]/(df[c1] + 0.0001)
        df[f'i{c1}m{c2}'] = df[c1]-df[c2]
        df[f'i{c2}m{c1}'] = df[c2]-df[c1]
        df[f'i{c1}p{c2}'] = df[c1]*df[c2]

def add_log_interaction_(df, cols):
    for c1, c2 in itertools.combinations(cols, 2):
        s1 = df[c1] if c1 != 'va02' else df.va02 - df.va02.min()
        s2 = df[c2] if c2 != 'va02' else df.va02 - df.va02.min()

        df[f'il{c2}s{c1}'] = np.log1p(s2)- np.log1p(s1)
        df[f'il{c1}p{c2}'] = np.log1p(s1)+ np.log1p(s2)
        df[f'il{c1}m{c2}'] = np.log1p(s1)* np.log1p(s2)
        df[f'il{c1}d{c2}'] = np.log1p(s1)/(np.log1p(s2)+0.0001)
        df[f'il{c2}d{c1}'] = np.log1p(s1)/(np.log1p(s2)+0.0001)

def target_encode_(df, cols, fold_col='fold', alpha=5, add_random=False, rmean=0, rstd=0.1):
    # https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study
    prior = df['ad_periods'].mean()
    for col in tqdm(cols):
        #kwargs = {f'{col}_target_freq': 'count',
        #          f'{col}_target_mean': 'mean',
        #          f'{col}_target_median': 'median', 
        #          f'{col}_target_std': 'std'}
        kwargs = {f'{col}_target_mean': 'mean'}

        for fold_ix in df['fold'].unique():
            n_samples = df.loc[df.fold != fold_ix].shape[0]
            nrows_cat = df.loc[df.fold != fold_ix].groupby(col)[col].count()
            enc = df.loc[(df.fold != fold_ix)].groupby(col)['ad_periods'].agg(**kwargs)
            for enc_c in enc.columns:
                __val_mask = (df.fold==fold_ix)
                if alpha > 0:
                    enc[enc_c] = (enc[enc_c] * nrows_cat + prior*alpha)/(nrows_cat + alpha)
                df.loc[__val_mask, enc_c] = df.loc[__val_mask, col].map(enc[enc_c]).fillna(prior)
                if add_random:
                    df.loc[__val_mask, enc_c] += normal(loc=rmean, scale=rstd, size=df.loc[__val_mask].shape[0])

def flag_target_encoding_(df, cols, fold_col='fold'):
    flag_te_cols = ["flag_min", "flag_max", "flag_cnt", "flag_sum", "flag_mean", "flag_std"]
    df[flag_te_cols] = np.nan
    for fold_ix in df[fold_col].unique():
        rows = []
        for c in cols:
            _df = df[(df[c]==1)&(df.fold!=fold_ix)]
            _min = _df['ad_periods'].min()
            _max = _df['ad_periods'].max()
            _cnt = _df[c].count()
            _sum = _df['ad_periods'].sum()
            _mean = _df['ad_periods'].mean()
            _std = _df['ad_periods'].std()
            rows.append([_min, _max, _cnt, _sum, _mean, _std])

        vals = df[df.fold==fold_ix][cols].values @ np.array(rows)
        df.loc[df[fold_col]==fold_ix, flag_te_cols] = vals

def flag_mean_encode_(df, fold_col='fold'):
    va_cols = [c for c in df.columns if c.startswith('va')]
    # columns with only two values (0, 1)
    uniq = df[va_cols].apply(lambda s: s.nunique())
    flag_cols = uniq[uniq==2].index.tolist()
    print("flag_cols:", flag_cols)

    flag_te_cols = ["flag_mean"]
    df[flag_te_cols] = np.nan
    for fold_ix in df[fold_col].unique():
        rows = []
        for c in flag_cols:
            _df = df[(df[c]==1)&(df[fold_col]!=fold_ix)]
            _cnt = _df[c].count()
            _sum = _df['ad_periods'].sum()
            _mean = _df['ad_periods'].mean()
            _std = _df['ad_periods'].std()
            _kurtosis = _df['ad_periods'].kurtosis()
            rows.append([_mean])
            #rows.append([_cnt, _sum, _mean, _std, _kurtosis])
            #rows.append([_cnt, _sum, _mean, _std, _kurtosis])
        vals = df[df[fold_col]==fold_ix][flag_cols].values @ np.array(rows)
        df.loc[df[fold_col]==fold_ix, flag_te_cols] = vals / df[df[fold_col]==fold_ix].shape[0]

#def prep_discrete(seed=270, n_splits=5):
#    rs = np.random.RandomState(seed)
#
#    dataroot = Path("./data")
#    dtrain = pd.read_csv(dataroot/'train.csv')
#    dtest = pd.read_csv(dataroot/'test.csv')
#
#    do_folding_(dtrain, 'fold', rs, n_splits=n_splits)
#
#    va_cols = [c for c in dtrain.columns if c.startswith('va')]
#    uniq = dtrain[[c for c in dtrain.columns if c.startswith('va')]].apply(lambda s: s.nunique())
#    flag_cols = uniq[uniq==2].index.tolist()
#    cols = [c for c in va_cols if c not in flag_cols]
#    print(cols)
#
#    enc = KBinsDiscretizer(n_bins=30, encode='ordinal')
#    dtrain_binned = enc.fit_transform(dtrain[cols])
#    dtrain[cols] = dtrain_binned
#
#    dtrain['cat1'] = dtrain[[f'dum_1_{i}' for i in range(1,6)]].values.nonzero()[1]
#    dtrain['cat2'] = dtrain[[f'dum_2_{i}' for i in range(1,12)]].values.nonzero()[1]
#    dtrain['cat3'] = dtrain[[f'dum_3_{i}' for i in range(1,4)]].values.nonzero()[1]
#    dtrain['cat4'] = dtrain[[f'dum_4_{i}' for i in range(1,15)]].values.nonzero()[1]
#
#    #me_cols = va_cols + ['cat1', 'cat2', 'cat3', 'cat4']
#    #target_encode_(dtrain, me_cols, alpha=5, add_random=False)
#    #dtrain.drop(me_cols, axis=1, inplace=True)
#
#    dtrain.drop([c for c in dtrain.columns if c.startswith('dum_')], axis=1, inplace=True)
#    torch.save(dtrain, 'prep/vanilla_binned.pt')

def prep(seed, dataroot, split):
    rs = np.random.RandomState(seed)

    dataroot = Path(dataroot)
    dtrain = pd.read_csv(dataroot/'train.csv')
    dtest = pd.read_csv(dataroot/'test.csv')

    torch.save(dtrain, 'prep/vanilla.pt')
    #sample_sub = pd.read_csv(dataroot/'sample_submission.csv')
    
    do_folding_(dtrain, 'fold', rs, n_splits=5)

    df = pd.concat([dtrain, dtest], axis=0)
    df.reset_index(drop=True, inplace=True)
    df.loc[:dtrain.shape[0],'split'] = 'train'
    df.loc[dtrain.shape[0]:,'split'] = 'test'

    df.loc[df.va06==12341234, 'va06'] = df.loc[df.va06!=12341234, 'va06'].mean()
    df.va06 = df.va06.clip(upper=0.45e6)
    df.loc[df.va07==2147483647, 'va07'] = df.loc[df.va07!=2147483647, 'va07'].mean()
    df['flag_va28'] = (df.va28 > 16000).astype('int')
    df['flag_va25_0'] = (df.va25==0).astype('int')
    df['va02'] = (df.va02/0.29221196 - 0.140799).round()

    # no contribution
    df.drop('va11', axis=1, inplace=True)

    # revert back dummy encoded to categorical
    df['cat1'] = df[[f'dum_1_{i}' for i in range(1,6)]].values.nonzero()[1]
    df['cat2'] = df[[f'dum_2_{i}' for i in range(1,12)]].values.nonzero()[1]
    df['cat3'] = df[[f'dum_3_{i}' for i in range(1,4)]].values.nonzero()[1]
    df['cat4'] = df[[f'dum_4_{i}' for i in range(1,15)]].values.nonzero()[1]

    #df.drop([c for c in df.columns if c.startswith('dum_')], axis=1, inplace=True)

    # extrac categorical (definetely helps) (super helpful)
    mask0 = df.va27<75000
    mask1 = (df.va27>75000)&(df.va27<120000)
    mask2 = df.va27>120000
    df.loc[mask0, 'flag_va27'] = 0
    df.loc[mask1, 'flag_va27'] = 1
    df.loc[mask2, 'flag_va27'] = 2

    uniq = df[[c for c in df.columns if c.startswith('va')]].apply(lambda s: s.nunique())
    flag_cols = uniq[uniq==2].index.tolist()

    df.va25 = df.va25.round(-1) # not helpful

    cols = [c for c in df.columns if c.startswith('va') and c not in flag_cols]
    lcols = ['l'+c for c in cols if c != 'va02']
    df[lcols] = df[[c for c in cols if c != 'va02']].apply(lambda s: np.log1p(s))
    df['lva02'] = np.log1p(df['va02']-df.va02.min())
    
    add_interaction_(df, cols)
    add_log_interaction_(df, cols)

    s = df[[c for c in df.columns if c not in ['ad_periods','fold']]].isna().sum()

    cols_to_keep = ['ilva01pva03','ilva01dva25','ilva25dva01','ilva01mva32','ilva01mva33','ilva05sva02','ilva02pva23','ilva07sva03','ilva13sva03','ilva03dva23','ilva23dva03','ilva03dva24','ilva24dva03','ilva04pva13','ilva04mva22','ilva04dva22','ilva22dva04','ilva04mva33','ilva06sva05','ilva07sva05','ilva05pva13','ilva27sva05','ilva05mva28','ilva05dva28','ilva28dva05','ilva05pva33','ilva13sva07','ilva07mva22','ilva23sva07','ilva23sva12','ilva32sva12','ilva22sva13','ilva24sva13','ilva13mva27','ilva13dva28','ilva28dva13','ilva22dva24','ilva24dva22','ilva23dva28','ilva28dva23','ilva32sva23','ilva24mva25','ilva24dva33','ilva33dva24','ilva25pva33','ilva27sva26','ilva26pva27','ilva26mva27','ilva26dva27','ilva27dva26','ilva26mva32','ilva26pva33','ilva33sva27','ilva27dva33','ilva33dva27']

    df.drop([c for c in df.columns if c.startswith('ilva') and c not in cols_to_keep], axis=1, inplace=True)
    cols_to_keep = ['iva01mva05','iva01pva33','iva02dva13','iva02dva22','iva02mva33','iva33mva02','iva05dva03','iva03mva05','iva12mva03','iva03mva13','iva03mva22','iva26mva03','iva32dva03','iva03pva32','iva23mva04','iva04pva24','iva04pva32','iva05mva27','iva27mva05','iva05pva27','iva23dva06','iva06pva24','iva28mva07','iva07mva32','iva22dva12','iva12pva23','iva24dva12','iva12mva27','iva12dva33','iva33mva12','iva22mva13','iva23mva13','iva13mva24','iva26mva13','iva13mva27','iva28mva13','iva13pva28','iva13mva32','iva33mva13','iva13pva33','iva22dva23','iva22mva24','iva24mva22','iva22pva24','iva32dva22','iva23dva28','iva24mva27','iva24pva27','iva28mva24','iva24dva33','iva27dva25','iva33dva25','iva26dva27','iva27dva26','iva26mva27','iva27mva26','iva26pva27','iva26mva33','iva33mva27']
    df.drop([c for c in df.columns if c.startswith('iva') and c not in cols_to_keep], axis=1, inplace=True)

    fn = f"prep/prep-{seed}-{split}.csv"
    print(f"writing to {fn}...")
    df.to_csv(fn, index=False)

if __name__=="__main__":
    parser = ArgumentParser(description="prep")
    parser.add_argument('--seed', default=43, type=int)
    parser.add_argument('--dataroot', type=str)
    parser.add_argument('--split', type=int, default=5)
    args = parser.parse_args()
    prep(args.seed, args.dataroot, args.split)
