import pathlib

import pandas as pd
from tqdm import tqdm
import random
import shutil
import numpy as np

# Define input directory
INPUT_DIR = pathlib.Path("data/raw/NUS_WIDE")

OUTPUT_DIR = pathlib.Path("data/processed") / f"NUS_WIDE"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)

metadata_path = INPUT_DIR / "ImageList" / "Imagelist.txt"

n_tags = 81

labels_info_dir = INPUT_DIR / "AllLabels"

n_imgs = sum(1 for _ in open(INPUT_DIR / "NUS-WIDE-urls.txt"))-1

minimum_frequency = 5000

rows = []

with open(metadata_path, "r") as input_file:
    for i, line in tqdm(enumerate(input_file), total=n_imgs, desc = "Getting images info..."):
        filename = line.split("\\")[-1].strip("\n")
        downloaded = (INPUT_DIR / "images" / filename).exists()
        rows.append([filename, downloaded])

metadata_df = pd.DataFrame(columns=["filename","downloaded"], data=rows)

labels_array = []

for file in labels_info_dir.iterdir():
    labels_array.append(np.loadtxt(file))

labels_array = np.array(labels_array).T

labels_array = labels_array.astype(int)

label_frequencies = np.sum(labels_array, axis = 0)

ordered_labels = list(np.argsort(label_frequencies)[::-1])

labels_array = labels_array[:,ordered_labels]

label_frequencies = label_frequencies[ordered_labels]

frequent_labels = (label_frequencies >= minimum_frequency)

labels = []

only_frequent_labels_array = labels_array[:,np.where(frequent_labels)[0]]

for row in only_frequent_labels_array:
    labels.append(str(row).replace("\n","").strip("[]"))

metadata_df["label"] = labels

label_name_df = pd.DataFrame()

if n_tags == 81:
    labels_info_file = INPUT_DIR / "Concepts81.txt"
elif n_tags == 1000:
    labels_info_file = INPUT_DIR / "TagList1k.txt"
else:
    print("Invalid number of tags.")
    quit()

with open(labels_info_file, "r") as input_file:
     label_name_df["name"] = input_file.readlines()

label_name_df["name"] = label_name_df["name"].apply(lambda x: x.strip("\n"))

label_name_df = label_name_df.loc[ordered_labels].reset_index()

label_name_df["id"] = label_name_df.index

label_name_df = label_name_df[["id","name"]]

label_name_df.to_csv(OUTPUT_DIR / f"label_names.txt", index=False)

label_name_df.loc[:21].to_csv(OUTPUT_DIR / "most_frequent_labels.txt", index=False)

#Filter only items with labels among the most frequent ones
metadata_df = metadata_df[(labels_array @ frequent_labels) > 0]

#Filter only downloaded items
metadata_df = metadata_df[metadata_df["downloaded"]==True]

full_df = metadata_df.copy()

full_df[["filename","label"]].to_csv(OUTPUT_DIR / f"complete_metadata.txt", header=False, index=False)

total_size = len(metadata_df)

#Set datafold sizes
train_size = 10500
query_size = 2100
val_size = 2100
database_size = total_size - train_size - query_size - val_size

datafolds = ["train"] * train_size + ["query"] * query_size + ["val"] * val_size + ["database"] * database_size

random.seed(0)
random.shuffle(datafolds)

metadata_df["datafold"] = datafolds

for datafold in ["train", "query", "val", "database"]:
    datafold_df = metadata_df[metadata_df["datafold"] == datafold].copy()

    datafold_df["image_path"] = datafold_df["filename"].apply(lambda x: f"data/processed/NUS_WIDE/images/{x}")

    for _, row in tqdm(datafold_df.iterrows(), total=len(datafold_df), desc=f"Saving images from {datafold}..."):
        img_source_path = INPUT_DIR / "images" / row["filename"]
        img_destination_path = pathlib.Path(row["image_path"])
        if img_destination_path.exists():
            continue
        else:
            shutil.copy(img_source_path, img_destination_path) 

    datafold_df.drop(columns=["filename", "datafold", "downloaded"], inplace=True)

    datafold_df = datafold_df[["image_path","label"]]

    # save filenames and label information
    datafold_df.to_csv(OUTPUT_DIR / f"{datafold}_metadata.txt", header=False, index=False)
