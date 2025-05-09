
selected_model="QS"
selected_experiment="rotation_experiment"

DBS=("CIFAR_10"
    "NUS_WIDE"
    "MS_COCO"
    "ImageNet"
    )

for selected_database in "${DBS[@]}"; do
    paths="$( find "models/$selected_model/$selected_database/$selected_experiment" -type d | grep "-no_cube$" | sort -R)"
    parallel --halt-on-error 2 --progress -j 10 'python src/models/ITQ/train.py -dir {1}' ::: "${paths[@]}"
done