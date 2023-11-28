import sys
import numpy as np
import time 
import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch

sys.path.insert(1, "src/")
from models.H2Q.lightning_module import LModule  # noqa: E402
from utils.architectures.FAST_HPP import Model

class FakeDataset(Dataset):
    def __init__(
        self,
        features,
        labels
    ):
        super().__init__()

        # load data
        self.features = features
        self.labels = labels

        # normalize features to sphere of radius sqrt{k}
        self.features /= np.linalg.norm(self.features, axis=1)[:, None]
        self.features *= np.sqrt(self.features.shape[1])
        
        self.data_len = len(self.features)
        self.n_classes = self.labels.shape[1]
        self.number_of_bits = self.features.shape[1]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx])


batch_size = 128
n_epochs = 300
n_classes = 20
n_trials = 10
ps = [.02, .02, .01, .95]

train_times = []
predict_times = []
nbitss = []
ns = []

for n in [100000, 1000000]:
    for nbits in [16, 32, 64]:
        
        class_size = n//n_classes
        features = np.vstack([ np.tile(np.random.normal(size=(1,nbits)), (class_size, 1)) for i in range(n_classes) ])
        features += .1 * np.random.normal(size=features.shape)
        features = features.astype(np.float32)
        labels = np.zeros((n, n_classes))
        for i in range(n_classes):
            labels[class_size*i:class_size*(i+1),i] += 1
        labels = labels.astype(np.float32)
        folds = np.random.choice([1,2,3,4], size=n, replace=True, p=ps)

        for _ in range(n_trials):
            nbitss.append(nbits)
            ns.append(n)

            ts = time.time()

            train_dataset = FakeDataset(features[folds == 1,:], labels[folds == 1,:])
            val_dataset = FakeDataset(features[folds == 2,:], labels[folds == 2,:])
            query_dataset = FakeDataset(features[folds == 3,:], labels[folds == 3,:])
            database_dataset = FakeDataset(features[folds == 4,:], labels[folds == 4,:])


            # define dataloaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1)

            # seed the experiment
            pl.seed_everything(0, workers=True)

            # init model
            model = LModule(
                Model(number_of_bits = train_dataset.number_of_bits),
                loss="L2",
                learning_rate=0.01
            )

            # get device to use
            trainer = pl.Trainer(
                max_epochs=n_epochs,
                accelerator="cpu",
                val_check_interval=1.0,
                logger=False
            )

            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            train_times.append(time.time() - ts)
            ts = time.time()

            # predict data
            model.eval()

            datasets = [train_dataset, val_dataset, query_dataset, database_dataset]
            datafolds = ["train", "val", "query", "database"]

            for dataset, datafold in zip(datasets, datafolds):
                datafold_features = model(torch.tensor(dataset.features, device=model.device)).detach().numpy()
                datafold_hashes = np.sign(datafold_features)

            predict_times.append(time.time() - ts)

pd.DataFrame({
    "train_times": train_times,
    "predict_times": predict_times,
    "nbits": nbitss,
    "n": ns
}).to_csv("experiments/times.csv")