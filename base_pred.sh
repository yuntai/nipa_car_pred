seed=43
splits=5
dataset=prep/prep-$seed-$splits.csv

#python train_pred.py -mlgb --mode base_pred -fbase --load lgb_43_base_150 --dataset $dataset 
#python train_pred.py -mlgb --mode base_pred -fint --load lgb_43_int_120 --dataset $dataset 
#python train_pred.py -mlgb --mode base_pred -flogint --load lgb_43_logint_150 --dataset $dataset 

#python train_pred.py -msvr --mode base_pred -fbase --load svr_43_base --dataset $dataset
python train.py -msvr --mode base_pred -fint --load svr_43_int --dataset $dataset
#python train_pred.py -msvr --mode base_pred -flogint --load svr_43_logint --dataset $dataset

#python train.py -mxgb --mode base_pred -fbase --load xgb_43_base --dataset $dataset
#python train.py -mxgb --mode base_pred -fint --load xgb_43_int --dataset $dataset
#python train.py -mxgb --mode base_pred -flogint --load xgb_43_logint --dataset $dataset

#python train_pred.py -mrf --mode base_pred -fbase --load rf_43_base --dataset $dataset
#python train_pred.py -mrf --mode base_pred -fint --load rf_43_int --dataset $dataset
#python train_pred.py -mrf --mode base_pred -flogint --load rf_43_logint --dataset $dataset
