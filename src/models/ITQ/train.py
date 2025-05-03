# ADAPTED FROM
# https://github.com/TreezzZ/ITQ_PyTorch/blob/master/itq.py


import sys
import numpy as np
import pathlib
import json
import scipy as sp
import torch

sys.path.insert(1, "src/")
from utils.general_utils import build_args  # noqa: E402
from utils.hashing_utils import save_sign_hashes

'''
ITQ solver class.
'''

torch.manual_seed(0)

def train_itq(train_features, centered = False, normalized=False, max_iter=50):
    """    
        Args
        train_features(torch.Tensor): Training data.
        max_iter(int): Number of iterations.
        device(torch.device): GPU or CPU.
        topk(int): Calculate top k data points map.

    Returns
        checkpoint(dict): Checkpoint.
    """

    nbits = train_features.shape[-1]

    # Initialization
    R = np.random.randn(nbits, nbits)
    [U, _, _] = np.linalg.svd(R)
    R = U[:,:nbits]

    if centered:
        center = np.mean(train_features, axis=0)
    else:
        center = np.zeros(nbits)

    V = train_features-center

    if normalized:
        V = norm_features(V, val="col")

    # Training
    for _ in range(max_iter):
        V_tilde = V @ R
        B = np.sign(V_tilde)
        R, _ = sp.linalg.orthogonal_procrustes(V, B)


    # Return rotation and center
    return R, center

def norm_features(features, val="col"):
    if val=="col":
            val = np.sqrt(features.shape[-1])
    return features / np.linalg.norm(features, axis=1)[:, np.newaxis] * val

args = build_args("src/models/ITQ/hparams.json", stage="train")

QS_DIR = pathlib.Path(args.dir)
hparams = json.load(open(QS_DIR / "hparams.json", "r"))
dir_name = QS_DIR.name

if args.normalized:
    dir_name = "-r_norm" + dir_name
if args.center:
    dir_name = "-r_center" + dir_name

ITQ_DIR = pathlib.Path("models/ITQ") / hparams["database"] / hparams["experiment_name"] / dir_name
ITQ_DIR.mkdir(parents=True, exist_ok=True)

train_features = np.load(QS_DIR / "train-features.npy")
R, center = train_itq(train_features, centered=args.center, normalized=args.normalized)

for fold in ["train", "query", "val", "database"]:
    features = np.load(QS_DIR / f"{fold}-features.npy")
    new_features = (features-center) @ R
    np.save(ITQ_DIR / f"{fold}-features.npy", new_features)
    save_sign_hashes(new_features, ITQ_DIR / f"{fold}-hashes.tsv")

hparams["r_center"] = args.center
hparams["r_norm"] = args.normalized
json.dump(hparams, open(ITQ_DIR / "hparams.json", "w"))
