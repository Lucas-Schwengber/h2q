#More information on NUS-WIDE
#https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html

import pathlib
import urllib.request
import patoolib
import zipfile
import pandas as pd
from tqdm import tqdm

def download(url, output_path):
    urllib.request.urlretrieve(url, output_path)

def unzip(input_path, output_dir, format = "zip"):
    if format=="rar":
        patoolib.extract_archive(input_path, outdir=output_dir)
    elif format=="zip":
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        print('Invalid format')

# def download_image(img_url, img_path):


TAGS_URL = "https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS_WID_Tags.zip"
CONCEPT_URL = "https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/ConceptsList.zip"
IMAGE_URL = "https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE-urls.rar"
IMAGE_LIST_URL = "https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/ImageList.zip"
LABELS_GT_LIST_URL = "https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/Groundtruth.zip"


# Make output directory if missing
OUTPUT_DIR = pathlib.Path("data/raw/NUS_WIDE")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# download(TAGS_URL, OUTPUT_DIR / "NUS_WID_Tags.zip")
# unzip(OUTPUT_DIR / "NUS_WID_Tags.zip", OUTPUT_DIR)

# download(CONCEPT_URL, OUTPUT_DIR / "ConceptsList.zip")
# unzip(OUTPUT_DIR / "ConceptsList.zip", OUTPUT_DIR)

download(LABELS_GT_LIST_URL, OUTPUT_DIR / "Groundtruth.zip")
unzip(OUTPUT_DIR / "Groundtruth.zip", OUTPUT_DIR)

download(IMAGE_URL, OUTPUT_DIR / "NUS-WIDE-urls.rar")

#Need to find a way to automatically unrar rar file
#unzip(OUTPUT_DIR / "NUS-WIDE-urls.rar", OUTPUT_DIR, format = "rar")

size = "Large"

size_index = {"Large":2, "Middle":3, "Small": 4}

#Download images
(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)

n_imgs = sum(1 for _ in open(OUTPUT_DIR / "NUS-WIDE-urls.txt"))

with open(OUTPUT_DIR / "NUS-WIDE-urls.txt", "r") as input_file:
    for i, line in tqdm(enumerate(input_file), total=n_imgs-1, desc = "Downloading images..."):
        if i == 0:
            continue
        else:
            line_info = line.split(" ")
            line_info = [info for info in line_info if info != ""]
            filename = line_info[0].split("\\")[-1]
            img_url = line_info[size_index[size]]
            downloaded = (OUTPUT_DIR / "images" / filename).exists()
            if downloaded:
                continue
            else:
                try:
                    download(img_url, OUTPUT_DIR / "images" / filename)
                except:
                    print(f"Could not download image {filename}")
