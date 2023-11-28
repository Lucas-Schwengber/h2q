import sys
import numpy as np
import torch
from pytorch_lightning import LightningModule, Callback
from torch.nn import CosineEmbeddingLoss, Tanh
from torch.optim import SGD, Adam, RMSprop

sys.path.insert(1, "src/")
from utils.losses import DSH, DPSH, DHN, DCH, WGLHH, HashNet, HyP2_pair, HyP2_proxy, CEL, HSWD  # noqa: E402
from utils.eval_utils import mAP_at

# define the LightningModule
class LModule(LightningModule):
    def __init__(
        self,
        model,
        loss,
        optimizer,
        learning_rate,
        weight_decay,
        penalty,
        number_of_classes,
        no_cube,
        similar_probability=0.5,
        L2_penalty=0,
        HSWD_penalty=0,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.number_of_classes = number_of_classes
        self.penalty = penalty
        self.L2_penalty = L2_penalty
        self.HSWD_penalty = HSWD_penalty

        self.val_hashes = []
        self.val_features = []
        self.val_labels = []

        self.using_hashnet = False
        self.balanced = False
        self.proxies = False
        
        self.scheduler_gamma = 0
        self.scheduler_period = 0

        self.batch_iter = 0

        self.val_k = None
        self.val_h = None

        self.no_cube = no_cube
        self.activation = Tanh()

        if loss == "DSH":
            self.loss = DSH(separation=2*self.model.number_of_bits, quantization_penalty=penalty)
            self.scheduler_gamma = 0.4
            self.scheduler_period = 20
        elif loss == "DPSH":
            self.loss = DPSH(quantization_penalty=penalty)
        elif loss == "DHN":
            self.loss = DHN(quantization_penalty=penalty)
        elif loss == "DCH":
            self.loss = DCH(quantization_penalty=penalty, p=similar_probability)
            self.val_h = 2
        elif loss == "WGLHH":
            self.loss = WGLHH(quantization_penalty=penalty, p=similar_probability)
            self.val_h = 2
        elif loss == "HashNet":
            self.using_hashnet = True
            self.hn_period = 200
            self.hn_power = 0.5
            self.loss = HashNet(p=similar_probability)
            self.no_cube = True
        elif loss == "HyP2":
            self.proxies = torch.nn.Parameter(
                torch.zeros(size=(self.number_of_classes, self.model.number_of_bits))
            )
            torch.nn.init.kaiming_normal_(self.proxies, mode = 'fan_out')
            codetable = np.genfromtxt("data/codetable.csv", delimiter=",")
            separation = codetable[self.model.number_of_bits][int(np.ceil(np.log2(self.number_of_classes)))]
            self.loss = HyP2_pair(separation = separation)
            self.proxy_loss = HyP2_proxy(separation = separation)
            self.scheduler_gamma = 0.5
            self.scheduler_period = 10
        elif loss == "CEL":
            codetable = np.genfromtxt("data/codetable.csv", delimiter=",")
            separation = codetable[self.model.number_of_bits][int(np.ceil(np.log2(self.number_of_classes)))]
            self.loss = CEL(separation = separation)
    
    def training_step(self, batch, batch_idx):
        X, label = batch
        Z = self.model(X)

        if not self.no_cube:
            Z = self.activation(Z)

        if self.using_hashnet:
            scale = (1 + self.batch_iter//self.hn_period)**self.hn_power
            Z = self.activation(scale*Z)
        
        loss = self.loss(Z, label)

        if self.L2_penalty > 0:
            loss += self.L2_penalty * torch.mean(torch.sum( (Z - torch.sign(Z))**2, dim = -1))
        
        if self.HSWD_penalty > 0:
            loss += self.HSWD_penalty * HSWD(Z)

        if torch.is_tensor(self.proxies):
            loss = self.penalty * loss + self.proxy_loss(Z, label, torch.tanh(self.proxies))     

        self.log("train_loss/batch", loss)
        self.batch_iter+=1
        return loss

    def validation_step(self, batch, batch_idx):
        X, label = batch

        Z = self.model(X)
        H = torch.sign(Z)

        self.val_hashes.append(H)
        self.val_features.append(Z)
        self.val_labels.append(label)

    def on_validation_epoch_end(self):
        hashes = torch.vstack(self.val_hashes).cpu().numpy()
        features = torch.vstack(self.val_features).cpu().numpy()
        labels = torch.vstack(self.val_labels).cpu().numpy()

        self.val_hashes = []
        self.val_features = []
        self.val_labels = []

        # new mAP
        mAPs = []
        for _ in range(5):
            size = 1000
            if size > hashes.shape[0]:
                size = hashes.shape[0]
            subsample = np.random.choice(hashes.shape[0], size=size, replace=False)
            vs = np.random.rand(size)
            q_is = subsample[vs <= .1]
            r_is = subsample[vs > .1]
            query_hashes = hashes[q_is, :]
            retrieval_hashes = hashes[r_is, :]
            query_features = features[q_is, :]
            retrieval_features = features[r_is, :]
            query_labels = labels[q_is, :]
            retrieval_labels = labels[r_is, :]
            mAPs.append(mAP_at(
                query_hashes,
                retrieval_hashes,
                query_features,
                retrieval_features,
                query_labels,
                retrieval_labels,
                h=self.val_h,
                k=self.val_k
            ))

        if self.model_has_nan():
            val_mAP = 0.0
        else:     
            val_mAP = np.mean(mAPs)

        self.logger.experiment.add_scalar("val_mAP/epoch", val_mAP, self.current_epoch)
        self.log("val_mAP", val_mAP, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        ps = [
            {"params": self.model.feature_layers.parameters(), "lr": self.learning_rate},
            {"params": self.model.hash_layer.parameters(), "lr": 10*self.learning_rate}
        ]
        if torch.is_tensor(self.proxies):
            ps.append({"params": self.proxies, "lr": 0.001})

        if self.optimizer == "adam":
            optimizer = Adam(ps, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            optimizer = SGD(ps, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        elif self.optimizer == "rmsprop":
            optimizer = RMSprop(ps, lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.scheduler_period > 0 and self.optimizer == "sgd":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size = self.scheduler_period,
                gamma = self.scheduler_gamma
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, data):
        # x is the image and y is its label
        x, y = data
        return self.model.forward(x), y
    
    def model_has_nan(self):
        return (sum([torch.sum(p.isnan()) for p in self.model.parameters()]) > 0).item()



        

