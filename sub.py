import torch
import pandas as pd

def sub():
    df = torch.load('prep/dfull.pt')
    m = torch.load('model.pt')
    dtest = df[train_sz:]

    test_pred = m.predict(dtest)
            
    pred_fn = 'sub/prediction.csv'
    sub = pd.read_csv('data/sample_submission.csv')
    sub['ad_periods'] = test_pred
    sub.to_csv(pred_fn, index=False)

if __name__ == '__main__':
    sub()
