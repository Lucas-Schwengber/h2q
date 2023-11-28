LRS=(
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
 
parallel --progress --halt-on-error 2 -j $2 'python3 src/models/ADSH/main.py -exp rotation_experiment -db ' $1 ' -lr 0.00001 -nbits {1} -bs 64 -ep 3 -mi 50 --gpu {2} -seed {2} -qss 5000 --architecture ' $3 ::: "${NBS[@]}" ::: "${SEEDS[@]}" 
