# cb fold run
model=xgb

seed=43
splits=5
dataset=prep/prep-$seed-$splits.csv

#feat="int"
#br=1902
#for fold in 0 1 2 3 4; do
#    name="${model}_${seed}_${feat}_${fold}"
#    echo "fitting ${feat}-${fold} $name"
#	python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --num_boost_round $br --fold $fold --save "$name"
#done
#
#feat="base"
#br=3054
#for fold in 0 1 2 3 4; do
#    name="${model}_${seed}_${feat}_${fold}"
#    echo "fitting ${feat}-${fold} $name"
#	python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --num_boost_round $br --fold $fold --save $name
#done

feat="logint"
br=2423
for fold in 0 1 2 3 4; do
    name="${model}_${seed}_${feat}_${fold}"
    echo "fitting ${feat}-${fold} $name"
	python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --num_boost_round $br --fold $fold --save $name
done
