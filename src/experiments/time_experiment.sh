#! /bin/bash
#SBATCH --mail-user=lucas.schwengber@berkeley.edu
#SBATCH --mail-type=ALL

# Train DPSH 4 times logging the time
# Then train the rotation

cd ~/h2q/h2q

source H2Q_env/bin/activate

ep=100
nw=4

rm -f 'time_experiment.txt'

for i in {1..4}
do
for nbits in 16 32 48 64
do

#fake run to start cache memory
python3 src/models/QS/train.py -exp time_experiment -db CIFAR_10 -loss DPSH -nw $nw -lr 0.00001 -nbits $nbits -bs 128 -ep $ep -pt 20 -seed $i -penalty 0 -no_skip -arch CNNF_alexnet -wd 0.0005

printf "\nsimilarity + penalty \nseed: ${i} \nnbits: ${nbits} \nstart: $(date +%s) "

cat >> time_experiment.txt << EOF
training + penalty
seed: ${i}
nbits: ${nbits}
start: $(date +%s)
EOF

for pen in 0 0.01 0.1 1
do
rm -rf models
python3 src/models/QS/train.py -exp time_experiment -db CIFAR_10 -loss DPSH -nw $nw -lr 0.00001 -nbits $nbits -bs 128 -ep $ep -pt 20 -seed $i -penalty $pen -no_skip -arch CNNF_alexnet -wd 0.0005

printf "finish lambda=${pen}: $(date +%s)"

cat >> time_experiment.txt << EOF
"finish lambda=${pen}: $(date +%s)"
EOF

done

printf "finish: $(date +%s)"

cat >> time_experiment.txt << EOF
finish: $(date +%s)
EOF

printf "\nprediction + penalty \nseed:  ${i} \nnbits: ${nbits} \nstart: $(date +%s) "

cat >> time_experiment.txt << EOF
predicting + penalty
seed: ${i}
nbits: ${nbits}
start: $(date +%s)
EOF

python3 src/models/QS/predict.py -exp time_experiment -db CIFAR_10 -loss DPSH -nw $nw -lr 0.00001 -nbits $nbits -bs 128 -ep $ep -pt 20 -seed $i -penalty 1 -no_skip -arch CNNF_alexnet -wd 0.0005

printf "finish: $(date +%s)"

cat >> time_experiment.txt << EOF
finish: $(date +%s)
EOF

rm -r models

printf "\nsimilarity + H2Q \nseed: ${i} \nnbits: ${nbits} \nstart: $(date +%s)"

cat >> time_experiment.txt << EOF
training + H2Q
seed: ${i}
nbits: ${nbits}
start: $(date +%s)
EOF

python3 src/models/QS/train.py -exp time_experiment -db CIFAR_10 -loss DPSH -nw $nw -lr 0.00001 -nbits $nbits -bs 128 -ep $ep -pt 20 -seed $i -penalty 0 -no_skip -arch CNNF_alexnet -wd 0.0005 -no_cube
python3 src/models/QS/predict.py -exp time_experiment -db CIFAR_10 -loss DPSH -nw $nw -lr 0.00001 -nbits $nbits -bs 128 -ep $ep -pt 20 -seed $i -penalty 0 -no_skip -arch CNNF_alexnet -wd 0.0005 -no_cube -dfs train val 
python3 src/models/H2Q/train.py -dir 'models/QS/CIFAR_10/time_experiment/-loss=DPSH-nbits='$nbits'-trf=imagenet-arch=CNNF_alexnet-seed='$i'-bs=128-ep='$ep'-pt=20-lr=1e-05-wd=0.0005-optim=adam-penalty=0.0-L2_penalty=0.0-HSWD_penalty=0.0-no_cube' -loss L2 -lr 0.1 -bs 128 -ep 300 -no_skip -no_pred

printf "finish: $(date +%s)"

cat >> time_experiment.txt << EOF
finish: $(date +%s)
EOF

printf "\npredict + H2Q \nseed: ${i} \nnbits: ${nbits} \nstart: $(date +%s)"

cat >> time_experiment.txt << EOF
prediction + H2Q
seed: ${i}
nbits: ${nbits}
start: $(date +%s)
EOF

python3 src/models/QS/predict.py -exp time_experiment -db CIFAR_10 -loss DPSH -nw $nw -lr 0.00001 -nbits $nbits -bs 128 -ep $ep -pt 20 -seed $i -penalty 0 -no_skip -arch CNNF_alexnet -wd 0.0005 -no_cube -dfs train val database query
python3 src/models/H2Q/predict.py -dir 'models/QS/CIFAR_10/time_experiment/-loss=DPSH-nbits='$nbits'-trf=imagenet-arch=CNNF_alexnet-seed='$i'-bs=128-ep='$ep'-pt=20-lr=1e-05-wd=0.0005-optim=adam-penalty=0.0-L2_penalty=0.0-HSWD_penalty=0.0-no_cube' -loss L2 -lr 0.1 -bs 128 -ep 300 -no_skip

printf "finish: $(date +%s)"

cat >> time_experiment.txt << EOF
finish: $(date +%s)
EOF

rm -rf models

done
done
