# ADAPTED FROM
# https://github.com/twistedcubic/learn-to-hash/blob/master/itq.py

import sys
import numpy as np
import pathlib
import json

sys.path.insert(1, "src/")
from utils.general_utils import build_args  # noqa: E402
from utils.hashing_utils import save_sign_hashes

'''
ITQ solver class.
'''

def itq_learn(X, center, perform_svd):
    """
    Runs ITQ... or hopes to
    X is the nxd data matrix
    """
    nbits = X.shape[1]

    # Zero-center points
    s = (1./X.shape[0]) * np.sum(X,0)
    if not center:
        s *= 0
    X = X - s
    
    # Preliminary dimension reduction
    if perform_svd:
        A2, _, _ = np.linalg.svd(np.dot(X.transpose(), X))
        W = A2[:,:nbits]
    else:
        W = np.eye(nbits)
    V = np.dot(X, W)
    
    # Initialize random rotation
    R = np.random.randn(nbits, nbits)
    Y1, _, _ = np.linalg.svd(R)
    R = Y1[:, :nbits]

    # Optimize in iterations
    for _ in range(50):
        tildeV = np.dot(V, R)
        B = np.ones(tildeV.shape)
        B[tildeV < 0] = -1
        Z = np.dot(B.transpose(), V)
        U2, T, U1 = np.linalg.svd(Z)
        R = np.dot(U1, U2.transpose())

    return s, W, R


args = build_args("src/models/ITQ/hparams.json", stage="train")

EMB_DIR = pathlib.Path(args.dir)
hparams = json.load(open(EMB_DIR / "hparams.json", "r"))
dir_name = EMB_DIR.name
if args.svd:
    dir_name = "-r_svd" + dir_name
if args.center:
    dir_name = "-r_center" + dir_name
ITQ_DIR = pathlib.Path("models/ITQ") / hparams["database"] / hparams["experiment_name"] / dir_name
ITQ_DIR.mkdir(parents=True, exist_ok=True)

train_features = np.load(EMB_DIR / "train-features.npy")
s, W, R = itq_learn(train_features, args.center, args.svd)

for fold in ["train", "query", "val", "database"]:
    features = np.load(EMB_DIR / f"{fold}-features.npy")
    tildeV = np.dot(np.dot(features - s, W), R)
    new_features = np.dot(features, R - s)
    np.save(ITQ_DIR / f"{fold}-features.npy", new_features)
    save_sign_hashes(new_features, ITQ_DIR / f"{fold}-hashes.tsv")

hparams["r_center"] = args.center
hparams["r_svd"] = args.svd
json.dump(hparams, open(ITQ_DIR / "hparams.json", "w"))