import pathlib
import shutil
from tqdm import tqdm

def copy_file(source_path, destination_path):
    if destination_path.exists():
        pass
    else:
        shutil.copy(source_path, destination_path)

input_datafolds = ["database","test","train"]

#The raw split information must be collected from the repository:
#https://github.com/thuml/HashNet

#And placed at:
#data/raw/ImageNet/hashnet/HashNet/pytorch/data/imagenet/database.txt
#data/raw/ImageNet/hashnet/HashNet/pytorch/data/imagenet/train.txt
#data/raw/ImageNet/hashnet/HashNet/pytorch/data/imagenet/test.txt

#Only run the script below after you've done the steps above.

input_dir = pathlib.Path("data/raw/ImageNet/hashnet/HashNet/pytorch/data/imagenet")

assert (input_dir / "database.txt").exists() and (input_dir / "train.txt").exists() and (input_dir / "test.txt").exists(), "Raw split information not avaliable. See README.md"

output_dir = pathlib.Path("data/processed/ImageNet_test")

images_output_dir = output_dir / "images"

output_dir.mkdir(parents=True, exist_ok=True)

(images_output_dir).mkdir(parents=True, exist_ok=True)

original_path_database = "/home/caozhangjie/run-czj/dataset/imagenet/image"
original_path_val = "/home/caozhangjie/run-czj/dataset/imagenet/val_image"
new_path_database = "data/raw/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"
new_path_val = "data/raw/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"

with open(input_dir / "test.txt","r") as input_file, open(output_dir / "val_metadata.txt","w") as val_file, open(output_dir / "query_metadata.txt","w") as query_file:
    lines = input_file.readlines()
    line_count = 0
    for line in lines:
        splited_line = line.split(" ")
        file_path = pathlib.Path(splited_line[0])
        labels = " ".join(splited_line[1:])
        source_file_path = str(file_path).replace(original_path_val, new_path_val)
        new_file_path = images_output_dir / source_file_path.split("/")[-1]
        copy_file(source_file_path,new_file_path)

        converted_row = ",".join([str(new_file_path),labels])

        if line_count % 2 == 0:
            query_file.write(converted_row)
        else:
            val_file.write(converted_row)

        line_count += 1

with open(input_dir / "train.txt","r") as input_file, open(output_dir / "train_metadata.txt","w") as train_file:
    lines = input_file.readlines()
    line_count = 0
    for line in lines:
        splited_line = line.split(" ")
        file_path = pathlib.Path(splited_line[0])
        file_prefix = file_path.name.split("_")[0]
        labels = " ".join(splited_line[1:])
        new_path = f"data/raw/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/{file_prefix}"
        source_file_path = str(file_path).replace(original_path_database, new_path)
        new_file_path = images_output_dir / source_file_path.split("/")[-1]
        copy_file(source_file_path,new_file_path)

        converted_row = ",".join([str(new_file_path),labels])

        train_file.write(converted_row)
        
        line_count += 1

with open(input_dir / "database.txt","r") as input_file, open(output_dir / "database_metadata.txt","w") as database_file:
    lines = input_file.readlines()
    line_count = 0
    for line in lines:
        splited_line = line.split(" ")
        file_path = pathlib.Path(splited_line[0])
        file_prefix = file_path.name.split("_")[0]
        labels = " ".join(splited_line[1:])
        new_path = f"data/raw/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/{file_prefix}"
        source_file_path = str(file_path).replace(original_path_database, new_path)
        new_file_path = images_output_dir / source_file_path.split("/")[-1]
        copy_file(source_file_path,new_file_path)

        converted_row = ",".join([str(new_file_path),labels])

        database_file.write(converted_row)
        
        line_count += 1


