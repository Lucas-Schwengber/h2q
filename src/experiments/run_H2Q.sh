
selected_model="QS"
selected_experiment="rotation_experiment"

DBS=(
"CIFAR_10"
"MS_COCO"
"NUS_WIDE"
"ImageNet"
)

for selected_database in "${DBS[@]}"; do
    paths="$( find "models/$selected_model/$selected_database/$selected_experiment" -type d | grep ".*-penalty=0.0.*" | sort -R)"

    parallel --progress --slf experiments/machines -k -j 1 --joblog experiments/joblog --retries 10 'cd ~/dyna_hash; source ~/hash/bin/activate; python src/models/H2Q/train.py -dir {1} -loss bit_var_loss -lr 0.01 -bs 128 -ep 300 >> experiments/logs/$(hostname -s).txt' ::: "${paths[@]}"
    parallel --progress --slf experiments/machines -k -j 1 --joblog experiments/joblog --retries 10 'cd ~/dyna_hash; source ~/hash/bin/activate; python src/models/H2Q/train.py -dir {1} -loss L2 -lr 0.1 -bs 128 -ep 300 >> experiments/logs/$(hostname -s).txt' ::: "${paths[@]}"
    parallel --progress --slf experiments/machines -k -j 1 --joblog experiments/joblog --retries 10 'cd ~/dyna_hash; source ~/hash/bin/activate; python src/models/H2Q/train.py -dir {1} -loss L1 -lr 0.1 -bs 128 -ep 300 >> experiments/logs/$(hostname -s).txt' ::: "${paths[@]}"
    parallel --progress --slf experiments/machines -k -j 1 --joblog experiments/joblog --retries 10 'cd ~/dyna_hash; source ~/hash/bin/activate; python src/models/H2Q/train.py -dir {1} -loss min_entry -lr 0.1 -bs 128 -ep 300 >> experiments/logs/$(hostname -s).txt' ::: "${paths[@]}"

done