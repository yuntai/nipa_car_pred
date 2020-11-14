# cb fold run
model=lgb

seed=43
splits=5
dataset=prep/prep-$seed-$splits.csv

feat="int"
nl=120
br=28547
for fold in 0 1 2 3 4; do
    name="${model}_${seed}_${feat}_${nl}_${fold}"
    echo "fitting ${feat}-${fold} $name"
	python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --lr_decay true --num_boost_round $br --num_leaves $nl --fold $fold --save "$name"
done

feat="base"
nl=150
br=29891
for fold in 0 1 2 3 4; do
    name="${model}_${seed}_${feat}_${nl}_${fold}"
    echo "fitting ${feat}-${fold} $name"
	python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --lr_decay true --num_boost_round $br --num_leaves $nl --fold $fold --save $name
done

feat="logint"
nl=150
br=23075
for fold in 0 1 2 3 4; do
    name="${model}_${seed}_${feat}_${nl}_${fold}"
    echo "fitting ${feat}-${fold} $name"
	python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --lr_decay true --num_boost_round $br --num_leaves $nl --fold $fold --save $name
done
