#!/bin/bash

#To limit number of cores per process:
#https://stackoverflow.com/questions/55746872/how-to-limit-number-of-cpus-used-by-a-python-script-w-o-terminal-or-multiproces

#source src/utils/trap_errors.sh

if (( $# < 2 ))
then
    echo './src/eval/eval_map.sh <model_name>'
    exit 1
fi

selected_model="$1"
selected_database="$2"
selected_experiment="$3"
paths="$( find "models/$selected_model/$selected_database/$selected_experiment" -type d )"

if  [ -z "$4" ]
then
jobs=2
else
jobs=$4
fi

c_p_j=4

mkdir -p "experiments/logs/eval_map"
echo "number_of_predictions=$( echo "$paths" | wc -l )"
echo
echo "Evaluating (stdout & stderr in experiments/logs/eval_map/)"
start_time=$SECONDS

echo "$paths" | parallel --halt-on-error 2 --progress -j $4 \
     'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python src/eval/eval_map.py -p {} &>> experiments/logs/eval_map/{#}.job'

elapsed=$(( SECONDS - start_time ))
echo
eval "echo Total eval time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"