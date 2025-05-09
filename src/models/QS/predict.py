import importlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

sys.path.insert(1, "src/")
from models.QS.lightning_module import LModule  # noqa: E402
from utils.general_utils import build_args, get_model_name, stringfy_model_args  # noqa: E402
from utils.hashing_utils import save_sign_hashes  # noqa: E402
from utils.datasets import VectorizedDataset  # noqa: E402

torch.multiprocessing.set_sharing_strategy('file_system')

model="QS"

# load arguments from the json file
args = build_args("src/models/QS/hparams.json", stage="predict")

# Generate model name
model_name = get_model_name(args,model)

# load the model and its parameters
database = args.database
experiment_name = args.experiment_name
model_dir = pathlib.Path("models/QS") / database / experiment_name / model_name

status_file_training = model_dir / "status=training.out"
status_file_interrupted = model_dir / "status=interrupted.out"
status_file_finished = model_dir / "status=finished.out"

training_started = status_file_training.exists()
training_interrupted = status_file_interrupted.exists()
already_finished = status_file_finished.exists()

hparams_path = model_dir / "hparams.json"

if not already_finished:
    print(f"Missing trained model for {model_name}.")
    sys.exit()

# get model output dir
output_model_dir = stringfy_model_args("src/models/QS/hparams.json", args, stage="predict", ignore=["-exp", "-db"])
OUTPUT_DIR = pathlib.Path("models/QS") / database / experiment_name / output_model_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

datafolds_to_predict = ["val", "query", "database", "train"]

if not args.no_skip and already_finished:
    prediction_suffix = "-hashes.tsv"

    skip_datafold = []

    for datafold in datafolds_to_predict:
        prediction_path = OUTPUT_DIR / (datafold+prediction_suffix)
        skip_datafold.append(prediction_path.exists())

if not args.no_skip and (all(skip_datafold)) and already_finished:
    print("Skipping predictions for:")
    print(model_dir)

else:
    # import the model
    Model = getattr(importlib.import_module(f"utils.architectures.{args.architecture}"), "Model")

    # load dataloaders
    dataloaders = []
    for datafold in args.datafolds:
        datafold_dataset = VectorizedDataset(args.database, datafold, load_matches=False)
        datafold_dataloader = DataLoader(
            datafold_dataset, batch_size=32, shuffle=False, num_workers=args.num_workers
        )
        dataloaders.append(datafold_dataloader)

    model = LModule.load_from_checkpoint(
        next(model_dir.glob("best-*")),
        model=Model(number_of_bits=args.number_of_bits),
        loss=args.loss,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        penalty=args.penalty,
        number_of_classes=datafold_dataset.n_classes,
        no_cube=args.no_cube,
        L2_penalty=args.L2_penalty,
        HSWD_penalty=args.HSWD_penalty,
    )
    model.eval()

    #save hparams
    with (OUTPUT_DIR / "hparams.json").open("w") as f:
        hparams = vars(args)
        hparams["model"] = "QS"
        hparams["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        f.write(json.dumps(hparams, indent=True))

    # initialize trainer for prediction
    if torch.cuda.device_count() > 0:
        trainer = pl.Trainer(
        	accelerator="gpu",
        	logger=False,
        	devices=[args.seed % torch.cuda.device_count()]
    		)
    else:
        trainer = pl.Trainer(
        	accelerator="cpu",
        	logger=False
    		)


    # predict
    for datafold, dataloader in zip(args.datafolds, dataloaders):
        prediction = trainer.predict(model, dataloader)
        transformed_data = np.vstack([p[0].numpy() for p in prediction])

        np.save(OUTPUT_DIR / f"{datafold}-features.npy", transformed_data)

        save_sign_hashes(transformed_data, OUTPUT_DIR / f"{datafold}-hashes.tsv")
