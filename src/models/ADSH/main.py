import pickle
import os
import logging
import torch
import time
import shutil
import pathlib
import json
import subprocess
import importlib
import copy 

import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms

from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys

sys.path.insert(1, "src/")
import models.ADSH.data_processing as dp
import models.ADSH.adsh_loss as al
import models.ADSH.subset_sampler as subsetsampler
import models.ADSH.calc_hr as calc_hr
import models.ADSH.cnn_model as cnn_model
import csv
from utils.general_utils import build_args, get_model_name  # noqa: E402

def _logging():
    logdir.mkdir(parents=True, exist_ok=True)
    global logger
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return

def _record():
    global record
    record = {}
    record['train loss'] = []
    record['iter time'] = []
    record['param'] = {}
    return

def encoding_onehot(target, nclasses=10):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot


def _dataset(database, datafolds):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dsets = {}
    nums = {}
    labels = {}

    for datafold in datafolds:

        dsets[datafold] = dp.ImageDatasetProcessing(
            f'data/processed/{database}/{datafold}_metadata.txt', transformations)
        nums[datafold] = len(dsets[datafold])

        labels[datafold] = torch.FloatTensor(dsets[datafold].label)

    return nums, dsets, labels

def calc_sim(database_label, train_label):
    S = (database_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    '''
    soft constraint
    '''
    r = S.sum() / (1-S).sum()
    S = S*(1+r) - r
    return S

def calc_loss(V, U, S, code_length, select_index, gamma, query_samples_size):
    num_train = V.shape[0]
    square_loss = (U.dot(V.transpose()) - code_length*S) ** 2
    V_omega = V[select_index, :]
    quantization_loss = (U-V_omega) ** 2
    loss = (square_loss.sum() + gamma * quantization_loss.sum()) / ( query_samples_size * num_train)
    return loss

def encode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    Z = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda())
        output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        Z[data_ind.numpy(), :] = output.cpu().data.numpy()
    return Z, B

def adjusting_learning_rate(optimizer, iter):
    update_list = [10, 20, 30, 40, 50]
    if iter in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10

def adsh_algo(database, code_length, datafolds_to_predict, model_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))

    '''
    parameter setting
    '''
    max_iter = args.max_iter
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = 5 * 10 ** -4
    query_samples_size = int(args.query_samples_size)
    gamma = args.gamma

    datafolds = ["train"] + ["database"] + datafolds_to_predict

    record['param']['args'] = args
    record['param']['description'] = '[Comment: learning rate decay]'
    logger.info(args)
    logger.info(code_length)
    logger.info(record['param']['description'])

    '''
    dataset preprocessing
    '''
 
    nums, dsets, labels = _dataset(database, datafolds)

    '''
    model construction
    '''

    training_set_size = nums["train"]

    model = getattr(importlib.import_module(f"utils.architectures.{args.architecture}"), "Model")(code_length)
    #model = cnn_model.CNNNet(args.architecture, code_length)
    model.to('cuda')
    adsh_loss = al.ADSHLoss(gamma, code_length, nums["train"])
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    V = np.zeros((nums["train"], code_length))

    model.train()
    for iter in range(max_iter):
        iter_time = time.time()
        '''
        sampling and construct similarity matrix
        '''
        select_index = list(np.random.permutation(range(nums["train"])))[0: query_samples_size]
        _sampler = subsetsampler.SubsetSampler(select_index)
        trainloader = DataLoader(dsets["train"], batch_size=batch_size,
                                 sampler=_sampler,
                                 shuffle=False,
                                 num_workers=4)
        '''
        learning deep neural network: feature learning
        '''
        sample_label = labels["train"].index_select(0, torch.from_numpy(np.array(select_index)))
        Sim = calc_sim(sample_label, labels["train"])
        U = np.zeros((query_samples_size, code_length), dtype=float)
        for epoch in range(epochs):
            for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader):
                batch_size_ = train_label.size(0)
                u_ind = np.linspace(iteration * batch_size, np.min((query_samples_size, (iteration+1)*batch_size)) - 1, batch_size_, dtype=int)
                train_input = Variable(train_input.cuda())

                output = model(train_input)
                S = Sim.index_select(0, torch.from_numpy(u_ind))
                U[u_ind, :] = output.cpu().data.numpy()

                model.zero_grad()
                loss = adsh_loss(output, V, S, V[batch_ind.cpu().numpy(), :])
                loss.backward()
                optimizer.step()
        adjusting_learning_rate(optimizer, iter)

        '''
        learning binary codes: discrete coding
        '''
        barU = np.zeros((nums["train"], code_length))
        barU[select_index, :] = U
        Q = -2*code_length*Sim.cpu().numpy().transpose().dot(U) - 2 * gamma * barU
        for k in range(code_length):
            sel_ind = np.setdiff1d([ii for ii in range(code_length)], k)
            V_ = V[:, sel_ind]
            Uk = U[:, k]
            U_ = U[:, sel_ind]
            V[:, k] = -np.sign(Q[:, k] + 2 * V_.dot(U_.transpose().dot(Uk)))
        iter_time = time.time() - iter_time
        loss_ = calc_loss(V, U, Sim.cpu().numpy(), code_length, select_index, gamma, query_samples_size)
        logger.info('[Iteration: %3d/%3d][Train Loss: %.4f]', iter, max_iter, loss_)
        record['train loss'].append(loss_)
        record['iter time'].append(iter_time)
    
    '''
    Save model
    '''

    torch.save(model.state_dict(), model_dir / "model.pt")

    '''
    training procedure finishes, evaluation
    '''
    model.eval()

    rB = V
    dataloader_train = DataLoader(dsets["train"], batch_size=1, shuffle=False, num_workers=4)
    encoded_train, _ = encode(model, dataloader_train, nums["train"], code_length)
    dataloader_database = DataLoader(dsets["database"], batch_size=1, shuffle=False, num_workers=4)
    encoded_database, dB = encode(model, dataloader_database, nums["database"], code_length)

    save_hashes(rB, model_dir / "train-hashes.tsv")
    np.save(model_dir / f"train-features.npy", encoded_train)

    save_hashes(dB, model_dir / "database-hashes.tsv")
    np.save(model_dir / f"database-features.npy", encoded_database)

    for datafold in datafolds_to_predict:
        dataloader = DataLoader(dsets[datafold], batch_size=1, shuffle=False, num_workers=4)
        encoded_datafold, qB = encode(model, dataloader, nums[datafold], code_length)
        np.save(model_dir / f"{datafold}-features.npy", encoded_datafold)
        save_hashes(qB, model_dir / f"{datafold}-hashes.tsv")
        map = calc_hr.calc_map(qB, dB, labels[datafold].numpy(), labels["database"].numpy())
        print(f"mAP for {datafold}: {map}")

def save_hashes(hashes, outpath):

    with open(outpath,"w", newline = '') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')

        for hash_vec in hashes:
            hash_list = [int(hash_entry) for hash_entry in hash_vec]
            file_row = hash_list
            writer.writerow(file_row)

global args, logdir

model="ADSH"

# load arguments from the json file
args = build_args("src/models/ADSH/hparams.json", stage="predict")

# Generate model name and create file if not exists
model_name = get_model_name(args, model, stage="predict")
database = args.database
model_dir = pathlib.Path(f"models/ADSH/{database}") / args.experiment_name / model_name

logdir = pathlib.Path("experiments/logs/ADSH") / args.experiment_name / model_name 

hparams_path = model_dir / "hparams.json"

already_trained = hparams_path.exists()

datafolds_to_predict = ["val","query"]

if not args.no_skip and already_trained:
    print("Skipping training for:")
    print(model_dir)

else:
    if model_dir.exists():
        shutil.rmtree(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    _logging()
    _record()
    adsh_algo(database, args.number_of_bits, datafolds_to_predict, model_dir)

    with (hparams_path).open("w") as f:
        hparams = vars(args)
        hparams["model"] = "ADSH"
        hparams["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        f.write(json.dumps(hparams, indent=True))