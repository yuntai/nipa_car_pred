
seed=43
splits=5
dataset=prep/prep-$seed-$splits.csv

#################################################################################################
model=rf
for feat in int base logint; do
    for fold in 0 1 2 3 4; do
        name="${model}_${seed}_${feat}_${fold}"
        echo "fitting ${feat}-${fold} $name"
        python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --fold $fold --save "$name"
    done

    name="${model}_${seed}_${feat}_all"
    python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --fold -1 --save "$name"
done


#################################################################################################
model=svr
for feat in int base logint; do
    for fold in 0 1 2 3 4; do
        name="${model}_${seed}_${feat}_${fold}"
        echo "fitting ${feat}-${fold} $name"
        python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --fold $fold --save "$name"
    done
    name="${model}_${seed}_${feat}_all"
    python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --fold -1 --save "$name"
done

#################################################################################################
model=lgb
feat="int"
nl=120
br=28547
for fold in 0 1 2 3 4; do
    name="${model}_${seed}_${feat}_${nl}_${fold}"
    echo "fitting ${feat}-${fold} $name"
	python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --lr_decay true --num_boost_round $br --num_leaves $nl --fold $fold --save "$name"
done
name="${model}_${seed}_${feat}_all"
python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --fold -1 --save "$name"

feat="base"
nl=150
br=29891
for fold in 0 1 2 3 4; do
    name="${model}_${seed}_${feat}_${nl}_${fold}"
    echo "fitting ${feat}-${fold} $name"
	python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --lr_decay true --num_boost_round $br --num_leaves $nl --fold $fold --save $name
done
name="${model}_${seed}_${feat}_all"
python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --fold -1 --save "$name"

feat="logint"
nl=150
br=23075
for fold in 0 1 2 3 4; do
    name="${model}_${seed}_${feat}_${nl}_${fold}"
    echo "fitting ${feat}-${fold} $name"
	python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --lr_decay true --num_boost_round $br --num_leaves $nl --fold $fold --save $name
done
name="${model}_${seed}_${feat}_all"
python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --fold -1 --save "$name"

#################################################################################################
model=xgb
feat="int"
br=1902
for fold in 0 1 2 3 4; do
    name="${model}_${seed}_${feat}_${fold}"
    echo "fitting ${feat}-${fold} $name"
	python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --num_boost_round $br --fold $fold --save "$name"
done
name="${model}_${seed}_${feat}_all"
python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --fold -1 --save "$name"

feat="base"
br=3054
for fold in 0 1 2 3 4; do
    name="${model}_${seed}_${feat}_${fold}"
    echo "fitting ${feat}-${fold} $name"
	python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --num_boost_round $br --fold $fold --save $name
done
name="${model}_${seed}_${feat}_all"
python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --fold -1 --save "$name"

feat="logint"
br=2423
for fold in 0 1 2 3 4; do
    name="${model}_${seed}_${feat}_${fold}"
    echo "fitting ${feat}-${fold} $name"
	python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --num_boost_round $br --fold $fold --save $name
done
name="${model}_${seed}_${feat}_all"
python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --fold -1 --save "$name"
