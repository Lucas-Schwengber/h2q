import pathlib

import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import random
import shutil
import csv

# Define input directory
INPUT_DIR = pathlib.Path("data/raw/MS_COCO")

metadata_df = pd.DataFrame(columns=["id", "filename", "label"]).set_index('id')
label_name_df = pd.DataFrame(columns=["id", "name"]).set_index('id')

n_labels = 90

for downloaded_datafold in ["train", "val"]:

    if downloaded_datafold == "val":
        METADATA_DIR = INPUT_DIR / "validation"
    else: 
        METADATA_DIR = INPUT_DIR / "train"

    metadata_path = METADATA_DIR / f"labels.json"

    with open(metadata_path,"r") as metadata_file:
        metadata = json.load(metadata_file)

    if downloaded_datafold == "train":
        for entry in tqdm(metadata['categories'], desc = f"Getting label names..."):
            label_name_df.loc[entry['id']] = [entry['name']]
        n_labels = label_name_df.index.max()

    for entry in tqdm(metadata['images'], desc = f"Getting metadata for {downloaded_datafold}..."):
        if (METADATA_DIR / "data" / entry['file_name']).exists():
            metadata_df.loc[entry['id']] = [entry['file_name'], np.zeros(n_labels).astype(int)]
    
    for entry in tqdm(metadata['annotations'], desc = f"Getting labels for {downloaded_datafold}..."):
        metadata_df.loc[entry['image_id'], "label"][entry["category_id"]-1] = 1

total_size = len(metadata_df)

#Set datafold sizes
train_size = 10000
query_size = 5000
val_size = 5000
database_size = total_size - train_size - query_size - val_size

datafolds = ["train"] * train_size + ["query"] * query_size + ["val"] * val_size + ["database"] * database_size

#Make random permutation of datafolds
random.seed(0)
random.shuffle(datafolds)

metadata_df["datafold"] = datafolds

OUTPUT_DIR = pathlib.Path("data/processed") / f"MS_COCO"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)

for datafold in ["train", "query", "val", "database"]:
    datafold_df = metadata_df[metadata_df["datafold"] == datafold].copy()

    datafold_df["image_path"] = datafold_df["filename"].apply(lambda x: f"data/processed/MS_COCO/images/{x}")
    datafold_df["label"] = datafold_df["label"].apply(lambda x: np.array2string(x).strip("[]").replace("\n",""))

    for _, row in tqdm(datafold_df.iterrows(), total=len(datafold_df), desc=f"Saving images from {datafold}..."):
        if "val" in row["filename"]:
            img_source_path = INPUT_DIR / "validation" / "data" / row["filename"]
        else:
            img_source_path = INPUT_DIR / "train" / "data" / row["filename"]

        img_destination_path = pathlib.Path(row["image_path"])
        if img_destination_path.exists():
            continue
        else:
            shutil.copy(img_source_path, img_destination_path) 

    datafold_df.drop(columns=["filename", "datafold"], inplace=True)

    datafold_df = datafold_df[["image_path","label"]]

    # save filenames and label information
    datafold_df.to_csv(OUTPUT_DIR / f"{datafold}_metadata.txt", header=False, index=False)

label_name_df.to_csv(OUTPUT_DIR / f"label_names.txt")





        


