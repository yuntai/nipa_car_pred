import models
import torch
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import gzip
import pickle
import time
import xgboost as xgb
from pathlib import Path

class SETTINGS:
    models = Path('models')

if __name__ == '__main__':

    parser = ArgumentParser(description="train")
    parser.add_argument('--seed', default=43, type=int)
    parser.add_argument('-m', '--model', default='lgb', choices=['h2o_gbm', 'cb', 'rf', 'et', 'xgb', 'lgb', 'lgbcl', 'tabnet','svr'])
    parser.add_argument('--dataset')
    parser.add_argument('-s', '--save', default='dev')
    parser.add_argument('--depth', default=16, type=int)
    parser.add_argument('--num_leaves', default=148, type=int)
    parser.add_argument('--mode', default='dev', choices=['dev', 'full', 'cv', 'hpsearch', 'fold', 'base_pred']),
    parser.add_argument('--resfn', default='res.csv')
    parser.add_argument('--clip_target', default=-1, type=int)
    parser.add_argument('--label', default="", type=str)
    parser.add_argument('--num_boost_round', default=50000, type=int)
    parser.add_argument('--lr_decay', default=False, type=bool)
    parser.add_argument('--verbose_eval', default=-1, type=int)
    parser.add_argument('--fold', default=-1, type=int)
    parser.add_argument('--load', default='', type=str)
    parser.add_argument('-f', '--feature_set', default='base', choices=['base', 'int', 'logint'])
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    s = df[df.split=='train'].isna().sum()
    assert s.sum()==0
    num_fold = df.fold.nunique()
    print("number of folds:", num_fold)
    assert args.fold == -1 or args.fold < fold_cnt

    dummy_cols = [c for c in df.columns if c.startswith('dum')]

    cols = [c for c in df.columns if c.startswith('va')]
    if args.feature_set == 'logint':
        cols += ['ilva01pva03','ilva01dva25','ilva25dva01','ilva01mva32','ilva01mva33','ilva05sva02','ilva02pva23','ilva07sva03','ilva13sva03','ilva03dva23','ilva23dva03','ilva03dva24','ilva24dva03','ilva04pva13','ilva04mva22','ilva04dva22','ilva22dva04','ilva04mva33','ilva06sva05','ilva07sva05','ilva05pva13','ilva27sva05','ilva05mva28','ilva05dva28','ilva28dva05','ilva05pva33','ilva13sva07','ilva07mva22','ilva23sva07','ilva23sva12','ilva32sva12','ilva22sva13','ilva24sva13','ilva13mva27','ilva13dva28','ilva28dva13','ilva22dva24','ilva24dva22','ilva23dva28','ilva28dva23','ilva32sva23','ilva24mva25','ilva24dva33','ilva33dva24','ilva25pva33','ilva27sva26','ilva26pva27','ilva26mva27','ilva26dva27','ilva27dva26','ilva26mva32','ilva26pva33','ilva33sva27','ilva27dva33','ilva33dva27']
    elif args.feature_set == 'int':
        cols += ['iva01mva05','iva01pva33','iva02dva13','iva02dva22','iva02mva33','iva33mva02','iva05dva03','iva03mva05','iva12mva03','iva03mva13','iva03mva22','iva26mva03','iva32dva03','iva03pva32','iva23mva04','iva04pva24','iva04pva32','iva05mva27','iva27mva05','iva05pva27','iva23dva06','iva06pva24','iva28mva07','iva07mva32','iva22dva12','iva12pva23','iva24dva12','iva12mva27','iva12dva33','iva33mva12','iva22mva13','iva23mva13','iva13mva24','iva26mva13','iva13mva27','iva28mva13','iva13pva28','iva13mva32','iva33mva13','iva13pva33','iva22dva23','iva22mva24','iva24mva22','iva22pva24','iva32dva22','iva23dva28','iva24mva27','iva24pva27','iva28mva24','iva24dva33','iva27dva25','iva33dva25','iva26dva27','iva27dva26','iva26mva27','iva27mva26','iva26pva27','iva26mva33','iva33mva27']
    cols = [c for c in cols if c not in ['fold', 'id', 'split']]

    cat_feats = ['cat1', 'cat2', 'cat3', 'cat4']
    flag_cols = [c for c in df.columns if c.startswith('flag')]
    cat_feats += flag_cols
    cat_feats = [c for c in cat_feats if c in df.columns]

    # flag cols themselves dummy
    if args.model in ['xgb', 'svr']:
        cols += dummy_cols
    else:
        cols += cat_feats

    cols += ['ad_periods']

    assert all(c in df.columns for c in cols)

    args.cols = cols

    print("cols=", cols)
    if args.model != 'xgb':
        print("cat_feats=", cat_feats)
        df.loc[:,cat_feats] = df[cat_feats].astype('int')

    if args.mode == 'full':
        x_df = df[df.split=='train'][cols]
        y_df = x_df.pop('ad_periods')
        m, tr_score, _ = models.__dict__[f"fit_{args.model}"](x_df, y_df, None, None, cat_feats, args)
        print("tr_score:", tr_score)
        arg.save = arg.save if arg.save else "full_model.pt"
        torch.save(m, f'{args.name}.pt')

    elif args.mode == 'base_pred':
        base_preds = []
        te_preds = []

        x_te_df = df[df.split=='test'][cols]
        x_te_df.pop('ad_periods')
        for fold_ix in range(num_fold):
            x_tr_df = df[(df.split=='train')&(df.fold==fold_ix)][cols]
            x_tr_df.pop('ad_periods')
            fn = f"models/{args.load}_{fold_ix}.m"
            print(f"loading {fn}...")
            with gzip.open(fn, 'rb') as inf:
                m = pickle.load(inf)
            if args.model == 'xgb':
                dtrain = xgb.DMatrix(data=x_tr_df)
                base_preds.append(m.predict(dtrain))
                dtest = xgb.DMatrix(data=x_te_df)
                te_preds.append(m.predict(dtest))
            else:
                base_preds.append(m.predict(x_tr_df))
                te_preds.append(m.predict(x_te_df))

        base_pred = np.concatenate(base_preds, axis=0)
        te_pred = np.stack(te_preds, axis=-1).mean(axis=-1)

        pd.DataFrame(data=base_pred, columns=[args.load]).to_csv('base/'+args.load+'_tr.csv', index=False)
        pd.DataFrame(data=te_pred, columns=[args.load]).to_csv('base/'+args.load+'_te.csv', index=False)

    elif args.mode in ['dev', 'hpsearch']:
        x_df = df[df.split=='train'][cols]
        y_df = x_df.pop('ad_periods')

        val_rate = 0.1
        x_tr, x_va, y_tr, y_va = train_test_split(x_df, y_df, test_size=val_rate, shuffle=True, random_state=args.seed)

        if args.mode == 'dev':
            m, tr_score, va_score = models.__dict__[f"fit_{args.model}"](x_tr, y_tr, x_va, y_va, cat_feats, args)
            print(tr_score, va_score)
            with open(args.resfn, 'a') as outf:
                outf.write(f"{args.label},{m.best_iteration},{tr_score},{va_score}\n")
            fn = f"models/model_dev-{va_score:.4f}.m"
            with gzip.open(fn, 'wb') as f:
                pickle.dump(m, f)
        else:
            if args.model == 'lgb':
                import hpopt
                hpopt.hpsearch_lgb(x_tr, y_tr, x_va, y_va, cat_feats, args)
            elif args.model == 'rf':
                models.hpsearch_rf(x_tr, y_tr, x_va, y_va, cat_feats, args)
            elif args.model == 'svr':
                models.hpsearch_svr(x_tr, y_tr, x_va, y_va, cat_feats, args)

    elif args.mode == 'fold':
        # when args.fold == -1, we use all the training data
        fn = SETTINGS.models/f"{args.save}.m"
        if not Path(fn).exists():
            _st = time.time()
            print(f"fold={args.fold} save={args.save}")
            fold_cnt = df.fold.nunique()
            if args.fold == -1:
                x_tr = df[(df.split=='train')][cols]
            else:
                # out of folder training
                x_tr = df[(df.split=='train') & (df.fold!=args.fold)][cols]
            y_tr = x_tr.pop('ad_periods')

            m, tr_score, _ = models.__dict__[f"fit_{args.model}"](x_tr, y_tr, None, None, cat_feats, args)
            et = time.time() - _st

            print(f"({et}) saving model to {fn}...")
            with gzip.open(fn, 'wb') as f:
                pickle.dump(m, f)
        else:
            print(f"{fn} already exists...")

    elif args.mode == 'cv':
        fold_cnt = df.fold.nunique()
        if args.nocv >= fold_cnt:
            r = range(fold_cnt)
        else:
            r = [args.nocv]
        tr_scores = []
        va_scores = []

        for fold_ix in r:
            print(f"fold_ix={fold_ix}")
            x_tr = df[(df.fold!=fold_ix)][cols]
            y_tr = x_tr.pop('ad_periods')

            x_va = df[(df.fold==fold_ix)][cols]
            y_va = x_va.pop('ad_periods')

            m, tr_score, va_score = models.__dict__[f"fit_{args.model}"](x_tr, y_tr, x_va, y_va, cat_feats, args)
            tr_scores.append(tr_score)
            va_scores.append(va_score)

        tr_scores = np.array(tr_scores)
        va_scores = np.array(va_scores)
        print(tr_scores.mean(), tr_scores.std(), va_scores.mean(), va_scores.std())
        with open(args.resfn, 'a') as outf:
            outf.write(f"{m.best_iteration},{tr_scores.mean()},{tr_scores.std()},{va_scores.mean()},{va_scores.std()}\n")
        
        torch.save(m, 'model_cv.pt')
