import sys
import pathlib
import numpy as np
import json
import shutil

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import torch

sys.path.insert(1, "src/")
from models.H2Q.lightning_module import LModule  # noqa: E402
from utils.architectures.FAST_HPP import Model
from utils.general_utils import build_args  # noqa: E402
from utils.hashing_utils import save_sign_hashes  # noqa: E402
from utils.datasets import EmbeddedDataset  # noqa: E402

# load arguments from the json file
args = build_args("src/models/H2Q/hparams.json", stage="train")

# load datasets
if "models" in args.directory:
    INPUT_DIR = pathlib.Path(args.directory)
else:
    INPUT_DIR = pathlib.Path("models/"+args.directory)

train_dataset = EmbeddedDataset(INPUT_DIR, "train")
val_dataset = EmbeddedDataset(INPUT_DIR, "val")
query_dataset = EmbeddedDataset(INPUT_DIR, "query")
database_dataset = EmbeddedDataset(INPUT_DIR, "database")

hparams_path = INPUT_DIR / "hparams.json"

if not hparams_path.exists():
    print(f"Missing hparams file for {INPUT_DIR}")
    sys.exit()

hparams = json.load(open(hparams_path, "r")) 
if args.batch_size > 0:
    batch_size = args.batch_size
else:
    batch_size = train_dataset.data_len

# define dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1)

# train model saving the checkpoints
database = hparams['database']
experiment_name = hparams['experiment_name']
model_name = f"-r_ep={args.epochs}-r_bs={args.batch_size}-r_lr={args.learning_rate}-r_loss={args.loss}"
model_name += args.directory.split("/")[-1]

OUTPUT_DIR = pathlib.Path(f"models/H2Q/{database}/{experiment_name}") / model_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

hparams_path = OUTPUT_DIR / "hparams.json"

checkpoints = [ckpt.name for ckpt in OUTPUT_DIR.glob("*.ckpt")]

status_file_training = OUTPUT_DIR / "status=training.out"
status_file_interrupted = OUTPUT_DIR / "status=interrupted.out"
status_file_finished = OUTPUT_DIR / "status=finished.out"

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
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # seed the experiment
    pl.seed_everything(hparams['seed'], workers=True)

    # create logger object
    logger_dir = pathlib.Path(f"experiments/lightning_logs/H2Q/{experiment_name}")
    logger_dir.mkdir(parents=True, exist_ok=True)
    logger = TensorBoardLogger(logger_dir, name=f"{database}-{model_name}")

    checkpoint_save_best = pl.callbacks.ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        save_top_k=1,
        monitor="val_loss",
        filename="best-{epoch}",
        mode="min",
    )

    # init model
    model = LModule(
        Model(number_of_bits = train_dataset.number_of_bits),
        loss=args.loss,
        learning_rate=args.learning_rate
    )

    # get device to use
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        val_check_interval=1.0,
        callbacks=[checkpoint_save_best],
        logger=logger
    )

    open(status_file_training,"w").close()

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # predict data
    model = LModule.load_from_checkpoint(
        next(OUTPUT_DIR.glob("best-*")),
        model=Model(number_of_bits = train_dataset.number_of_bits),
        loss=args.loss,
        learning_rate=args.learning_rate,
    )
    model.eval()

    datasets = [train_dataset, val_dataset, query_dataset, database_dataset]
    datafolds = ["train", "val", "query", "database"]

    for dataset, datafold in zip(datasets, datafolds):
        features = model(torch.tensor(dataset.features, device=model.device)).detach().cpu().numpy()
        np.save(OUTPUT_DIR / f"{datafold}-features.npy", features)
        save_sign_hashes(features, OUTPUT_DIR / f"{datafold}-hashes.tsv")

    if trainer.state.status == "finished":

        with (hparams_path).open("w") as f:
            hparams["r_lr"] = args.learning_rate
            hparams["r_loss"] = args.loss
            hparams["r_epochs"] = args.epochs
            hparams["r_batch_size"] = args.batch_size
            json.dump(hparams, open(hparams_path, "w"), indent=4)

        (status_file_training).rename(status_file_finished)

    elif trainer.state.status == "interrupted":

        (status_file_training).rename(status_file_interrupted)
    