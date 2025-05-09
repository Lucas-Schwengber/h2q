# Code resources companion for "Deep Hashing via Householder Quantization" (arXiv:2311.04207)

## Setup

Our experiments involve training CNNs several times (to account for different seeds, number of bits, architectures, databases and losses). Besides that, we also need to train rotations (via ITQ and also via H²Q) on the embeddings previously obtained, which fortunately can be done purely on CPU.

In our lab the CNNs were trained on a cluster with 4 x A100 GPUs and took around 3 days. The rotations were trained on CPU. The process of evaluating the metrics also makes intense usage of CPU and it was parallelized in CPU.

To start you must install our requirements, avaliable in <code>requirements.txt</code>. We recommend you to use a virtual environment and install the requirements as follows:

```shell
python -m venv H2Q_env
source H2Q_env/bin/activate
pip install -r requirements.txt
```

To re-initialize the environment if needed run:
```shell
source H2Q_env/bin/activate
```

## Initializing the directories

To start, create the following directories:
```shell
mkdir data
mkdir experiments
mkdir models
mkdir eval
```

The <code>data</code> directory is the one that will contain our four datasets (CIFAR_10, ImageNet, MS_COCO and NUS_WIDE). See more below.

The <code>experiments</code> directory will contain logs of experiments.

The <code>models</code> directory will contain the trained models weights and metadata.

The <code>eval</code> directory will contain the metrics of each trained model.


## Downloading and processing the datasets

The first step is to download and process the datasets. We have 4 datasets: CIFAR_10, ImageNet, MS_COCO and NUS_WIDE. The following code will download the raw datasets, CIFAR_10, MS_COCO and NUS_WIDE:
```shell
python src/data/download/CIFAR_10.py
python src/data/download/MS_COCO.py
python src/data/download/NUS_WIDE.py
```

To pre-process the three datasets above run:
```shell
python src/data/process/CIFAR_10.py
python src/data/process/MS_COCO.py
python src/data/process/NUS_WIDE.py
```

For ImageNet the procedure is slightly different. The download must be done via Kaggle. The instructions are in <code>src/data/download/ImageNet.py</code>.
For pre-processing, the raw information containing the splits must be collected from the HashNet repository https://github.com/thuml/HashNet and placed at:
```shell
data/raw/ImageNet/hashnet/HashNet/pytorch/data/imagenet/database.txt
data/raw/ImageNet/hashnet/HashNet/pytorch/data/imagenet/train.txt
data/raw/ImageNet/hashnet/HashNet/pytorch/data/imagenet/test.txt
```

After this, the pre-processing step should be done running:

```shell
python src/data/process/ImageNet.py
```

After running downloading and processing the datasets you can proceed to the experiments.

## Running a single experiment

We illustrate the full procedure to train, predict and evaluate a single hyperparameter combination of a given model.

First train running:
```shell
python src/models/QS/train.py -exp rotation_experiment -db CIFAR_10 -loss CEL -nw 4 -lr 0.00001 -nbits 16 -bs 128 -ep 2 -pt 20 -seed 0 -penalty 0.0 -L2_penalty 0.01 -arch CNNF_alexnet -wd 0.0005
```
Then predict running:
```shell
python src/models/QS/predict.py -exp rotation_experiment -db CIFAR_10 -loss CEL -nw 4 -lr 0.00001 -nbits 16 -bs 128 -ep 2 -pt 20 -seed 0 -penalty 0.0 -L2_penalty 0.01 -arch CNNF_alexnet -wd 0.0005
```
Finally evaluate by running:
```shell
python src/eval/eval_map.py -p "models/QS/CIFAR_10/rotation_experiment/-loss=CEL-nbits=16-trf=imagenet-arch=CNNF_alexnet-seed=0-bs=128-ep=2-pt=20-lr=1e-05-wd=0.0005-optim=adam-penalty=0.0-L2_penalty=0.01-HSWD_penalty=0.0"
```
One can then train a rotation on top of the embedding learned using:
```shell
python src/models/H2Q/train.py -dir "models/QS/CIFAR_10/rotation_experiment/-loss=CEL-nbits=16-trf=imagenet-arch=CNNF_alexnet-seed=0-bs=128-ep=2-pt=20-lr=1e-05-wd=0.0005-optim=adam-penalty=0.0-L2_penalty=0.01-HSWD_penalty=0.0"
```

And evaluate the results from the rotation:
```shell
python src/eval/eval_map.py -p "models/H2Q/CIFAR_10/rotation_experiment/-r_ep=150-r_bs=128-r_lr=1.0-r_loss=bit_var_loss-loss=CEL-nbits=16-trf=imagenet-arch=CNNF_alexnet-seed=0-bs=128-ep=2-pt=20-lr=1e-05-wd=0.0005-optim=adam-penalty=0.0-L2_penalty=0.01-HSWD_penalty=0.0"
```

## Running the experiments

Our experiments can be divided in four classes:
- $\lambda = 0$ without tanh activation;
- $\lambda > 0$ with tanh activation;
- $\lambda = 0.1$ with the HSWD penalty without tanh activation;
- $\lambda = 0$ without tanh activation + H²Q (with the 4 possible losses);
- $\lambda = 0$ without tanh activation + ITQ;

All the scripts run experiments in parallel using the parallel

The first 3 experiments involve training a CNN (alexnet or vgg16) are are quite costly. The last two can be performed in CPU since they use the embeddings trained in the first two experiments.

In the scripts below:
- <code>$nj</code> is the number of jobs used to parallel the experiments (adapt it to your setup);
- <code>$nw</code> is the number of workers used during the trainer of a single job;

To train the embeddings corresponding to the first 3 experiments run:

```shell
bash src/experiments/run_QS.sh CIFAR_10 $nj $nj*$nw CNNF_alexnet
bash src/experiments/run_QS.sh NUS_WIDE $nj $nj*$nw CNNF_alexnet
bash src/experiments/run_QS.sh ImageNet $nj $nj*$nw CNNF_alexnet
bash src/experiments/run_QS.sh MS_COCO $nj $nj*$nw CNNF_alexnet

bash src/experiments/run_QS.sh CIFAR_10 $nj $nj*$nw CNNF_vgg16
bash src/experiments/run_QS.sh NUS_WIDE $nj $nj*$nw CNNF_vgg16
bash src/experiments/run_QS.sh ImageNet $nj $nj*$nw CNNF_vgg16
bash src/experiments/run_QS.sh MS_COCO $nj $nj*$nw CNNF_vgg16

```

After training you can evaluate the results using the <code>eval_map.sh</code> script as follows:
```shell
bash src/eval/eval_map.sh QS CIFAR_10 rotation_experiment $nw
bash src/eval/eval_map.sh QS NUS_WIDE rotation_experiment $nw
bash src/eval/eval_map.sh QS ImageNet rotation_experiment $nw
bash src/eval/eval_map.sh QS MS_COCO rotation_experiment $nw
```

The ADSH benchmark can be trained using
```shell
bash src/experiments/run_ADSH.sh CIFAR_10 $nj CNNF_alexnet
bash src/experiments/run_ADSH.sh NUS_WIDE $nj CNNF_alexnet
bash src/experiments/run_ADSH.sh ImageNet $nj CNNF_alexnet
bash src/experiments/run_ADSH.sh MS_COCO $nj CNNF_alexnet

bash src/experiments/run_ADSH.sh CIFAR_10 $nj CNNF_vgg16
bash src/experiments/run_ADSH.sh NUS_WIDE $nj CNNF_vgg16
bash src/experiments/run_ADSH.sh ImageNet $nj CNNF_vgg16
bash src/experiments/run_ADSH.sh MS_COCO $nj CNNF_vgg16
```
and evaluated using
```shell
bash src/eval/eval_map.sh ADSH CIFAR_10 rotation_experiment $nw
bash src/eval/eval_map.sh ADSH NUS_WIDE rotation_experiment $nw
bash src/eval/eval_map.sh ADSH ImageNet rotation_experiment $nw
bash src/eval/eval_map.sh ADSH MS_COCO rotation_experiment $nw
```
for all databases.

To train the H²Q run (make sure to have a <code>experiments/machines</code> file with a list of machines to paralelize in CPU, if you don't have multiple machines you can adapt the script to run in a single machine):
```shell
bash experiments/run_H2Q.sh
```
and to evaluate the results run
```shell
bash src/eval/eval_map.sh H2Q CIFAR_10 rotation_experiment $nw
bash src/eval/eval_map.sh H2Q NUS_WIDE rotation_experiment $nw
bash src/eval/eval_map.sh H2Q ImageNet rotation_experiment $nw
bash src/eval/eval_map.sh H2Q MS_COCO rotation_experiment $nw
```
for all databases.

And to run all experiments using ITQ run
```shell
bash experiments/run_ITQ.sh $nw
```
and to evaluate the results run
```shell
bash src/eval/eval_map.sh ITQ CIFAR_10 rotation_experiment $nw
bash src/eval/eval_map.sh ITQ NUS_WIDE rotation_experiment $nw
bash src/eval/eval_map.sh ITQ ImageNet rotation_experiment $nw
bash src/eval/eval_map.sh ITQ MS_COCO rotation_experiment $nw
```
for all databases.

Finally, there is also an experiment verifying the times for training and prediction. You can run that experiment running
```shell
bash src/experiments/time_experiment.sh
```

## How to reproduce images and tables

After running all the commands of listed above, one can reproduce our images and tables with the <code>notebooks/images_and_tables.ipynb</code> notebook.
