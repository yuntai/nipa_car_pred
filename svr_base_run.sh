# svr fold run
model=svr

seed=43
splits=5
dataset=prep/prep-$seed-$splits.csv

for feat in int base logint; do
    for fold in 0 1 2 3 4; do
        name="${model}_${seed}_${feat}_${fold}"
        echo "fitting ${feat}-${fold} $name"
        python train.py -m$model --seed $seed --mode fold -f$feat --dataset $dataset --fold $fold --save "$name"
    done
    exit
done
