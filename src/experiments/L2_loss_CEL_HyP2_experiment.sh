jobs=4
c_p_j=16

NBS=(
'16' 
'32' 
'48' 
'64' 
) 
 
SEEDS=(
'0' 
'1' 
'2' 
'3' 
)

loss="HyP2"
db="ImageNet"
penalty="1.25"
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -exp rotation_experiment -db '$db' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$1' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -exp rotation_experiment -db '$db' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$1' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

loss="HyP2"
db="CIFAR_10"
penalty="1.25"
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -exp rotation_experiment -db '$db' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$1' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -exp rotation_experiment -db '$db' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$1' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

loss="HyP2"
db="MS_COCO"
penalty="1.25"
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -exp rotation_experiment -db '$db' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$1' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -exp rotation_experiment -db '$db' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$1' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

loss="HyP2"
db="NUS_WIDE"
penalty="0.75"
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -exp rotation_experiment -db '$db' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$1' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -exp rotation_experiment -db '$db' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$1' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 


DBS=(
"CIFAR_10"
"NUS_WIDE"
"MS_COCO"
"ImageNet"
)
loss="CEL"
penalty="0.0"
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -exp rotation_experiment -db {3} -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 128 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$1' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" ::: "${DBS[@]}"
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -exp rotation_experiment -db {3} -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 128 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$1' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" ::: "${DBS[@]}"
