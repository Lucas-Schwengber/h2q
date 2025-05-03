import argparse
import json
import pathlib
import sys

import torch
import pandas as pd
import numpy as np


def get_type(s):
    if s == "int":
        return int
    if s == "list":
        return list
    if s == "str":
        return str
    if s == "float":
        return float


def build_args(path, stage="train"):
    hparams = json.load(open(path))
    parser = argparse.ArgumentParser()

    for key in hparams:
        if stage in hparams[key]["stages"]:
            if "nargs" in hparams[key]:
                parser.add_argument(
                    key,
                    hparams[key]["fullname"],
                    default=hparams[key]["default"],
                    help=hparams[key]["help"],
                    type=get_type(hparams[key]["type"]),
                    nargs=hparams[key]["nargs"],
                )
            elif hparams[key]["type"] == "bool":
                if hparams[key]["default"]:
                    parser.add_argument(
                        key,
                        hparams[key]["fullname"],
                        action="store_false",
                        help=hparams[key]["help"],
                    )
                else:
                    parser.add_argument(
                        key,
                        hparams[key]["fullname"],
                        action="store_true",
                        help=hparams[key]["help"],
                    )
            else:
                parser.add_argument(
                    key,
                    hparams[key]["fullname"],
                    default=hparams[key]["default"],
                    help=hparams[key]["help"],
                    type=get_type(hparams[key]["type"]),
                )

    return parser.parse_args()


def stringfy_model_args(path, args, stage="train", ignore=[]):
    hparams = json.load(open(path))
    args_used = vars(args)

    result = ""

    for key in hparams:
        if hparams[key]["in_filename"] and stage in hparams[key]["stages"] and key not in ignore:
            if hparams[key]["type"] == "bool":
                # save bool only if true
                if args_used[hparams[key]["fullname"][2:]]:
                    result += key
            else:
                p_name = key
                p_type = hparams[key]["type"]
                p_value = args_used[hparams[key]["fullname"][2:]]
                if p_type == "list":
                    p_value = "-".join([str(v) for v in p_value])
                result += p_name + "=" + str(p_value)
    return result


def get_device():
    return "gpu" if torch.cuda.is_available() else "cpu"


def get_model_name(args, model, stage="train"):

    if model=="dyna_hash":

        if args.update_match_probability_method == "sine":
            model_name = stringfy_model_args(
                "src/models/dyna_hash/hparams.json", args, stage, ignore=["-ars", "-marmd", "-nmarmd", "-prb"]
            )

        elif args.update_match_probability_method == "adaptative":
            model_name = stringfy_model_args(
                "src/models/dyna_hash/hparams.json", args, stage, ignore=["-sp", "-prb"]
            )

        elif args.update_match_probability_method == "constant":
            model_name = stringfy_model_args(
                "src/models/dyna_hash/hparams.json", args, stage, ignore=["-ars", "-marmd", "-nmarmd", "-sp"]
            )
        else:
            model_name = stringfy_model_args("src/models/dyna_hash/hparams.json", args, stage, ignore=["-ars", "-marmd", "-nmarmd", "-sp", "-prb"])
    
    else:
        if pathlib.Path(f"src/models/{model}/hparams.json").exists():
            model_name = stringfy_model_args(f"src/models/{model}/hparams.json", args, stage, ignore=["-exp", "-db"])
        else:
            print("Invalid model.")
            model_name = None
    
    return model_name


def rename_hparam(model_group, new_name_short, old_name_short, new_name_long, old_name_long):
    MODEL_DIR = pathlib.Path("models") / model_group
    EVAL_DIR = pathlib.Path("eval") / model_group

    for database_dir in MODEL_DIR.iterdir():
        for model_folder in database_dir.iterdir():

            if (model_folder / "hparams.json").exists():

                with open(model_folder / "hparams.json", "r") as input_file:
                    hparams = json.load(input_file)
                
                if old_name_long in hparams:
                    hparams[new_name_long] = hparams.pop(old_name_long)

                with open(model_folder / "hparams.json", "w") as output_file:
                    output_file.write(json.dumps(hparams, indent=True))

            new_path = pathlib.Path(str(model_folder).replace(old_name_short,new_name_short))
            model_folder.rename(new_path)

    
    for database_dir in EVAL_DIR.iterdir():
        for model_folder in database_dir.iterdir():
                
            for datafold in ["val","query"]:
                if (model_folder / f"{datafold}-prediction_mAP.json").exists():
                    with open(model_folder / f"{datafold}-prediction_mAP.json", "r") as input_file:
                        hparams = json.load(input_file)

                    if old_name_long in hparams['hparams']:
                        hparams['hparams'][new_name_long] = hparams['hparams'].pop(old_name_long)

                    with open(model_folder / f"{datafold}-prediction_mAP.json", "w") as output_file:
                        output_file.write(json.dumps(hparams, indent=True))

                new_path = pathlib.Path(str(model_folder).replace(old_name_short,new_name_short))
                model_folder.rename(new_path)


#Function to load points, hashes and labels of a given trained model
def load_features_hashes_and_labels(model_dir, datafold = "all"):
    path = pathlib.Path(model_dir)
    hparams_path = path / "hparams.json"
    hparams = json.load(open(hparams_path, "r"))
    dataset = hparams["database"]
    DATA_DIR = pathlib.Path(f"data/processed/{dataset}")

    if datafold == "all":
        datafolds = ["train","val","query","database"]
    else:
        datafolds = [datafold]

    features = {}
    hashes = {}
    labels = {}

    for datafold in datafolds:
        hashes_path = path / f"{datafold}-hashes.tsv"
        features_path = path / f"{datafold}-features.npy"
        hashes[datafold] = np.loadtxt(hashes_path,comments='#',delimiter='\t').astype(int)
        metadata = pd.read_csv(DATA_DIR / f"{datafold}_metadata.txt", header=None, names=["file_path","labels"])
        labels[datafold] = np.array([label.split(" ") for label in metadata["labels"]]).astype(float)
        features[datafold] = np.load(features_path)

    if datafold == "all":
        return features, hashes, labels
    else:
        return features[datafold], hashes[datafold], labels[datafold]
    

def get_device_free():
    # i = np.argmax([torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())])
    i = np.random.choice(torch.cuda.device_count())
    return int(i)