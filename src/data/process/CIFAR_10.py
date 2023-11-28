import json
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import random

# Define input directory
INPUT_DIR = pathlib.Path("data/raw/CIFAR_10/cifar-10-batches-py")

# Open files with picke
meta_file = pickle.load(open(INPUT_DIR / "batches.meta", "rb"), encoding="bytes")
train_files = [pickle.load(open(INPUT_DIR / f"data_batch_{i}", "rb"), encoding="bytes") for i in range(1,6)]
test_file = pickle.load(open(INPUT_DIR / "test_batch", "rb"), encoding="bytes")

# make pandas df with entries data
filenames = [filename.decode("utf-8") for filename in sum([train_file[b"filenames"] for train_file in train_files],[]) + test_file[b"filenames"]]
raw_labels = [label for label in sum([train_file[b"labels"] for train_file in train_files],[]) + test_file[b"labels"]]
labels = [np.eye(1,10,raw_label)[0].astype(int) for raw_label in raw_labels]

total_size = len(labels)

datafolds = ["database"] * total_size

shuffled_indexes = [i for i in range(0,total_size)]

#Make random permutation of indexes
random.seed(0)
random.shuffle(shuffled_indexes)

#Build 'train', 'query' and 'val' datafold using fixed number of examples per class
for i in range(0,10):
    train_indexes_for_class = [j for j in shuffled_indexes if raw_labels[j]==i][:500]
    query_indexes_for_class = [j for j in shuffled_indexes if raw_labels[j]==i][500:600]
    val_indexes_for_class = [j for j in shuffled_indexes if raw_labels[j]==i][600:700]
    for index in train_indexes_for_class:
        datafolds[index] = "train"
    for index in query_indexes_for_class:
        datafolds[index] = "query"
    for index in val_indexes_for_class:
        datafolds[index] = "val"


df = pd.DataFrame(
    {
        "filename": filenames,
        "label": labels,
        "datafold": datafolds,
    }
)

# make dataframe with metadata linking label to label_name
label_map = pd.DataFrame(
        {
            "id": np.sort(np.unique(raw_labels)),
            "name": [name.decode("utf-8") for name in meta_file[b"label_names"]],
        }
    )

# join train and test data
data = np.vstack([train_file[b"data"] for train_file in train_files] + [test_file[b"data"]])

# data is flatten in a counterintuitive way here, we need to fix it
for i in range(data.shape[0]):
    data[i, :] = np.reshape(data[i, :], (32, 32, 3), order="F").transpose(1, 0, 2).flatten()

# generate cifar 10
OUTPUT_DIR = pathlib.Path("data/processed") / f"CIFAR_10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# save label map
label_map.to_csv(OUTPUT_DIR / "label_names.txt", index=False)

for datafold in ["train", "query", "val", "database"]:
    datafold_df = df[df["datafold"] == datafold].copy()

    datafold_df["image_path"] = datafold_df["filename"].apply(lambda x: f"data/processed/CIFAR_10/images/{x}")
    datafold_df["label"] = datafold_df["label"].apply(lambda x: np.array2string(x).strip("[]"))

    #save images
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    for index, row in tqdm(datafold_df.iterrows(), total=len(datafold_df), desc="Saving images: "):
        file_path = OUTPUT_DIR / "images" / row["filename"]
        if file_path.exists():
            continue
        else:
            Image.fromarray(np.reshape(data[index, :], (32, 32, 3))).save(file_path)

    datafold_df.drop(columns=["filename", "datafold"], inplace=True)

    datafold_df = datafold_df[["image_path","label"]]
  
    # save filename and label information
    datafold_df.to_csv(OUTPUT_DIR / f"{datafold}_metadata.txt", index=False, header=False)
