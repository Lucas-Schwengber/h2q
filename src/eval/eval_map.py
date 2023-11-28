import numpy as np
import argparse
import json
import sys
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(1, "src/")
from utils.eval_utils import mAP_at_many

def plot_hist(int_hashes, labels, figname):
    int_hashes = int_hashes.astype(int)
    u,c = np.unique(int_hashes, return_counts=True)
    h_to_use = u[c > np.max(c)/10].tolist()
    map_h = {}
    map_h[-1] = -1
    for i,h in enumerate(h_to_use):
        map_h[h] = i

    use_int_hashes = []
    use_labels = []
    for h,l in zip(int_hashes, labels):
        if h in h_to_use:
            use_int_hashes.append(map_h[h])
            use_labels.append(l)
        else:
            use_int_hashes.append(-1)
            use_labels.append(l)

    use_labels = np.array(use_labels)
    use_int_hashes = np.array(use_int_hashes)
    x = []
    for label in np.unique(use_labels):
        x.append( use_int_hashes[use_labels == label] )
    plt.hist(x, np.arange(-1, len(h_to_use)+1), stacked=True)
    plt.xticks([map_h[h]+.5 for h in map_h], [h for h in map_h], rotation=90)
    plt.savefig(figname)
    plt.clf()


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--path",
    type=str,
    required=True,
    help = "Path to predictions."
)
parser.add_argument(
    "-no_skip",
    "--no_skip",
    action='store_true',
    help = "Skip if predictions already exist."
)
parser.add_argument(
    "-make_plots",
    "--make_plots",
    action='store_true',
    help = "Make plots."
)

args = parser.parse_args()

path = pathlib.Path(args.path)
datafolds = ["query","val"]
hparams_path = path / "hparams.json"

if not hparams_path.exists():
    print(f"Missing predictions for {path}")
    sys.exit()
hparams = json.load(open(hparams_path, "r"))

dataset = hparams["database"]
DATA_DIR = pathlib.Path(f"data/processed/{dataset}")
out_folder = pathlib.Path(str(path).replace("models","eval"))
out_folder.mkdir(parents=True, exist_ok=True)
database_datafolds = ["database"]
database_hashes = None
database_labels = None

for datafold in database_datafolds:
    hashes_path = path / f"{datafold}-hashes.tsv"
    features_path = path / f"{datafold}-features.npy"

    if hashes_path.exists() and features_path.exists():
    
        datafold_metadata = pd.read_csv(DATA_DIR / f"{datafold}_metadata.txt", header=None, names=["file_path","labels"])
        datafold_labels = np.array([label.split(" ") for label in datafold_metadata["labels"]]).astype(int)
        datafold_hashes = np.loadtxt(hashes_path,comments='#',delimiter='\t').astype(int)
        datafold_features = np.load(features_path)
        if database_hashes is not None and database_labels is not None:
            database_hashes = np.concatenate((database_hashes, datafold_hashes), axis = 0)
            database_labels = np.concatenate((database_labels, datafold_labels), axis = 0)
            database_features = np.concatenate((database_features, datafold_features), axis = 0)
        else:
            database_hashes = datafold_hashes
            database_labels = datafold_labels
            database_features = datafold_features
            
    else:
        print(f"Missing predictions for {path}.")
        sys.exit()

for datafold in datafolds:

    out_path = out_folder / f"{datafold}-prediction_mAP.json"

    if out_path.exists() and not args.no_skip:
        print(f"Skipping eval for {datafold} on {path}.")
        continue
    else:
        print(f"Evaluating prediction for {datafold}")
        hashes_path = path / f"{datafold}-hashes.tsv"
        features_path = path / f"{datafold}-features.npy"

        if hashes_path.exists() and database_hashes is not None:
            query_hashes = np.loadtxt(hashes_path,comments='#',delimiter='\t').astype(int)
            query_metadata = pd.read_csv(DATA_DIR / f"{datafold}_metadata.txt", header=None, names=["file_path","labels"])
            query_labels = np.array([label.split(" ") for label in query_metadata["labels"]]).astype(int)
            query_features = np.load(features_path)

            k_values = [1000, 5000]

            mAPs = mAP_at_many(query_features, database_features, query_labels, database_labels, [], k_values)

            if args.make_plots:
                hash_powers = 2**np.arange(0, query_hashes.shape[1])
                label_powers = 2**np.arange(0, query_labels.shape[1])
                int_query_labels = query_labels @ label_powers
                int_database_labels = database_labels @ label_powers
                plot_hist(((query_hashes+1)/2) @ hash_powers, int_query_labels, out_folder / f"{datafold}-hashes.pdf")
                if datafold == datafolds[0]:
                    plot_hist(((database_hashes+1)/2) @ hash_powers, int_database_labels, out_folder / f"dataset-hashes.pdf")

            out_dict = {
                "mAP": mAPs['full'],
                "hparams": hparams,
                "output_path": str(out_path),
                "datafold": datafold
            }

            for k in k_values:
                out_dict[f"mAP_at_k={k}"] = mAPs[f"k={k}"]

            json.dump(out_dict, open(out_path, "w"), indent=4)
        else:
            print(f"{hashes_path} does not exists.")




