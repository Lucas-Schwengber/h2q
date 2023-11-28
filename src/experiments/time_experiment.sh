# Train HyP² 20 times logging the time
# Then train the rotation

touch 'experiments/time_experiment.txt'

for i in {1..20}
do
    for nbits in 16 32 48 64
    do
        echo -n $i $nbits "similarity" "$(date +%s)" >> 'time_experiment.txt'
        python3 src/models/PenaltyStrategies/train.py -exp time_experiment -db NUS_WIDE -loss HyP2 -nw 16 -lr 0.00001 -nbits $nbits -bs 128 -ep 100 -pt 20 -seed $i -penalty 0.5 -no_cube -no_skip -arch alexnet -wd 0.0005
        echo " $(date +%s)" >> 'experiments/time_experiment.txt'
    done
done

for i in {1..20}
do
    for nbits in 16 32 48 64
    do
        echo -n $i $nbits "predict" "$(date +%s)" >> 'time_experiment.txt'
        python3 src/models/PenaltyStrategies/predict.py -exp time_experiment -db NUS_WIDE -loss HyP2 -nw 16 -lr 0.00001 -nbits $nbits -bs 128 -ep 100 -pt 20 -seed $i -penalty 0.5 -no_cube -no_skip -arch alexnet -wd 0.0005
        echo " $(date +%s)" >> 'experiments/time_experiment.txt'
    done
done