
selected_model="PenaltyStrategies"
selected_experiment="rotation_experiment"

DBS=(
"CIFAR_10"
"MS_COCO"
"NUS_WIDE"
"ImageNet"
)

for selected_database in "${DBS[@]}"; do
    paths="$( find "models/$selected_model/$selected_database/$selected_experiment" -type d | grep ".*-penalty=0.0.*" | sort -R)"
    parallel --halt-on-error 2 --progress -j $1 'python src/models/ITQ/train.py -dir {1}' ::: "${paths[@]}"
done