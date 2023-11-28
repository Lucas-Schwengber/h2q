jobs=$2 
c_p_j=$(($3 / $2)) 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.5' 
'0.75' 
'1.0' 
'1.25' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py  -exp rotation_experiment -db ' $1 ' -loss HyP2 -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 100 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py  -exp rotation_experiment -db ' $1 ' -loss HyP2 -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 100 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.0' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py  -exp rotation_experiment -db ' $1 ' -loss WGLHH -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 64 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0001' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py  -exp rotation_experiment -db ' $1 ' -loss WGLHH -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 64 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0001' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.0' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py  -exp rotation_experiment -db ' $1 ' -loss HashNet -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 256 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py  -exp rotation_experiment -db ' $1 ' -loss HashNet -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 256 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.0' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py  -exp rotation_experiment -db ' $1 ' -loss DCH -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 256 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py  -exp rotation_experiment -db ' $1 ' -loss DCH -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 256 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.0' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py  -exp rotation_experiment -db ' $1 ' -loss DHN -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 64 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py  -exp rotation_experiment -db ' $1 ' -loss DHN -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 64 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.001' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py  -exp rotation_experiment -db ' $1 ' -loss WGLHH -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 64 -ep 100 -pt 20 -seed {4} -penalty {2}  -arch '$4' -wd 0.0001' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py  -exp rotation_experiment -db ' $1 ' -loss WGLHH -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 64 -ep 100 -pt 20 -seed {4} -penalty {2}  -arch '$4' -wd 0.0001' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.00001' 
'0.001' 
'0.1' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py  -exp rotation_experiment -db ' $1 ' -loss DCH -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 256 -ep 100 -pt 20 -seed {4} -penalty {2}  -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py  -exp rotation_experiment -db ' $1 ' -loss DCH -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 256 -ep 100 -pt 20 -seed {4} -penalty {2}  -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.00001' 
'0.001' 
'0.1' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py  -exp rotation_experiment -db ' $1 ' -loss DHN -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 64 -ep 100 -pt 20 -seed {4} -penalty {2}  -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py  -exp rotation_experiment -db ' $1 ' -loss DHN -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 64 -ep 100 -pt 20 -seed {4} -penalty {2}  -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.0' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py  -exp rotation_experiment -db ' $1 ' -loss CEL -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 128 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py  -exp rotation_experiment -db ' $1 ' -loss CEL -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 128 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.0' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py  -exp rotation_experiment -db ' $1 ' -loss DPSH -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 128 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py  -exp rotation_experiment -db ' $1 ' -loss DPSH -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 128 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'1.0' 
'0.1' 
'0.01' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py  -exp rotation_experiment -db ' $1 ' -loss DPSH -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 128 -ep 100 -pt 20 -seed {4} -penalty {2}  -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py  -exp rotation_experiment -db ' $1 ' -loss DPSH -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 128 -ep 100 -pt 20 -seed {4} -penalty {2}  -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.75' 
'1.25' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -HSWD_penalty 0.1 -exp rotation_experiment -db ' $1 ' -loss HyP2 -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 100 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -HSWD_penalty 0.1 -exp rotation_experiment -db ' $1 ' -loss HyP2 -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 100 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.0' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -HSWD_penalty 0.1 -exp rotation_experiment -db ' $1 ' -loss WGLHH -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 64 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0001' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -HSWD_penalty 0.1 -exp rotation_experiment -db ' $1 ' -loss WGLHH -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 64 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0001' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.0' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -HSWD_penalty 0.1 -exp rotation_experiment -db ' $1 ' -loss DCH -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 256 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -HSWD_penalty 0.1 -exp rotation_experiment -db ' $1 ' -loss DCH -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 256 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.0' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -HSWD_penalty 0.1 -exp rotation_experiment -db ' $1 ' -loss DHN -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 64 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -HSWD_penalty 0.1 -exp rotation_experiment -db ' $1 ' -loss DHN -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 64 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

LRS=(
'0.00001' 
) 
 
PENS=(
'0.0' 
) 
 
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
 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -HSWD_penalty 0.1 -exp rotation_experiment -db ' $1 ' -loss CEL -nw ' $c_p_j ' -lr {1} -nbits {3} -bs 128 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $2 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -HSWD_penalty 0.1 -exp rotation_experiment -db ' $1 ' -loss CEL -nw ' $c_p_j  ' -lr {1} -nbits {3} -bs 128 -ep 100 -pt 20 -seed {4} -penalty {2} -no_cube -arch '$4' -wd 0.0005' ::: "${LRS[@]}" ::: "${PENS[@]}" ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

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
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -exp rotation_experiment -db '$1' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$4' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -exp rotation_experiment -db '$1' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$4' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

loss="HyP2"
db="CIFAR_10"
penalty="1.25"
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -exp rotation_experiment -db '$1' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$4' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -exp rotation_experiment -db '$1' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$4' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

loss="HyP2"
db="MS_COCO"
penalty="1.25"
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -exp rotation_experiment -db '$1' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$4' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -exp rotation_experiment -db '$1' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$4' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 

loss="HyP2"
db="NUS_WIDE"
penalty="0.75"
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -exp rotation_experiment -db '$1' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$4' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -exp rotation_experiment -db '$1' -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 100 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$4' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" 


DBS=(
"CIFAR_10"
"NUS_WIDE"
"MS_COCO"
"ImageNet"
)
loss="CEL"
penalty="0.0"
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/train.py -exp rotation_experiment -db {3} -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 128 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$4' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" ::: "${DBS[@]}"
parallel --progress --halt-on-error 2 -j $jobs 'taskset --cpu-list $(( (({#}-1) % '$jobs'+1)*'$c_p_j' ))-$(( (({#}-1) % '$jobs'+1)*'$c_p_j' + '$c_p_j'-1 )) python3 src/models/PenaltyStrategies/predict.py -exp rotation_experiment -db {3} -loss '$loss' -nw ' $c_p_j ' -lr 0.00001 -nbits {1} -bs 128 -ep 100 -pt 20 -seed {2} -penalty '$penalty' -L2_penalty 0.01 -arch '$4' -wd 0.0005' ::: "${NBS[@]}" ::: "${SEEDS[@]}" ::: "${DBS[@]}"
