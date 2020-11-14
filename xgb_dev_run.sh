model=xgb
seed=43
dataset=prep/prep-$seed-5.csv

feat="int"
python train.py -m$model --seed $seed --mode dev -f$feat --dataset $dataset

feat="base"
python train.py -m$model --seed $seed --mode dev -f$feat --dataset $dataset

feat="logint"
python train.py -m$model --seed $seed --mode dev -f$feat --dataset $dataset
