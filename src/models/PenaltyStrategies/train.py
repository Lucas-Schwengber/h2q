import importlib
import pathlib
import shutil
import sys
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import subprocess
import torch
import json

sys.path.insert(1, "src/")
from models.PenaltyStrategies.lightning_module import LModule  # noqa: E402
from utils.general_utils import build_args, get_model_name  # noqa: E402
from utils.datasets import VectorizedDataset  # noqa: E402

torch.multiprocessing.set_sharing_strategy('file_system')

# load arguments from the json file
args = build_args("src/models/PenaltyStrategies/hparams.json", stage="train")

# Generate model name and create file if not exists
model_name = get_model_name(args, "PenaltyStrategies")
database = args.database
experiment_name = args.experiment_name
model_dir = pathlib.Path(f"models/PenaltyStrategies/{database}/{experiment_name}") / model_name

hparams_path = model_dir / "hparams.json"

status_file_training = model_dir / "status=training.out"
status_file_interrupted = model_dir / "status=interrupted.out"
status_file_finished = model_dir / "status=finished.out"

training_started = status_file_training.exists()
training_interrupted = status_file_interrupted.exists()
already_finished = status_file_finished.exists()

data_folds_to_predict = ["val","test"]

if not args.no_skip and already_finished:
    print("Skipping training for:")
    print(model_name)
elif not args.no_skip and training_started and args.soft_skip:
    print("Soft skipping training for:")
    print(model_name)
else:
    if model_dir.exists():
        shutil.rmtree(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    # seed the experiment
    pl.seed_everything(args.seed, workers=True)

    # import the nn model
    Model = getattr(importlib.import_module(f"utils.architectures.{args.architecture}"), "Model")

    # load train and val datasets
    train_dataset = VectorizedDataset(args.database, "train", load_matches=2)
    val_dataset = VectorizedDataset(args.database, "val", load_matches=False)

    # define dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # init model
    model = LModule(
        Model(number_of_bits=args.number_of_bits),
        loss=args.loss,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        penalty=args.penalty,
        number_of_classes=train_dataset.n_classes,
        similar_probability=train_dataset.get_similar_probability(),
        no_cube=args.no_cube,
        L2_penalty=args.L2_penalty,
        HSWD_penalty=args.HSWD_penalty,
    )
    
    # create logger object
    logger_dir = pathlib.Path(f"experiments/lightning_logs/{database}/{experiment_name}")
    logger_dir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(logger_dir, name=model_name)

    # train model saving the checkpoints
    checkpoint_save_all = pl.callbacks.ModelCheckpoint(
        dirpath=f"models/PenaltyStrategies/{database}/{experiment_name}/{model_name}",
        save_top_k=-1,
        every_n_epochs=args.patience // 2,
        monitor="val_mAP",
        filename="{epoch}-{val_mAP:.8f}",
    )
    checkpoint_save_best = pl.callbacks.ModelCheckpoint(
        dirpath=f"models/PenaltyStrategies/{database}/{experiment_name}/{model_name}",
        save_top_k=1,
        monitor="val_mAP",
        filename="best-{epoch}",
        mode="max",
    )

    if torch.cuda.device_count() != 0:
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu",
            devices=[args.seed % torch.cuda.device_count()],
            val_check_interval=1.0,
            callbacks=[
                checkpoint_save_all,
                checkpoint_save_best,
                EarlyStopping(monitor="val_mAP", mode="max", patience=args.patience, min_delta=0.05),
            ],
            logger=logger,
            num_sanity_val_steps=0
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="cpu",
            val_check_interval=1.0,
            callbacks=[
                checkpoint_save_all,
                checkpoint_save_best,
                EarlyStopping(monitor="val_mAP", mode="max", patience=args.patience, min_delta=0.05),
            ],
            logger=logger,
            num_sanity_val_steps=0
        )
        

    open(status_file_training,"w").close()
    
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if trainer.state.status == "finished":

        with (hparams_path).open("w") as f:
            hparams = vars(args)
            hparams["model"] = "PenaltyStrategies"
            hparams["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
            f.write(json.dumps(hparams, indent=True))

        (status_file_training).rename(status_file_finished)

    elif trainer.state.status == "interrupted":

        (status_file_training).rename(status_file_interrupted)
