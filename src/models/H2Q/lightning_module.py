import sys
from pytorch_lightning import LightningModule
from torch.optim import SGD

sys.path.insert(1, "src/")
from utils.losses import bit_var_loss, L2, min_entry, cos_sim, L1  # noqa: E402

# define the LightningModule
class LModule(LightningModule):
    def __init__(
        self,
        model,
        loss="bit_var_loss",
        learning_rate=1
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        if loss=="bit_var_loss":
            self.loss = bit_var_loss()
        if loss=="L2":
            self.loss = L2()
        if loss=="min_entry":
            self.loss = min_entry()
        if loss=="L1":
            self.loss = L1()
    
    def training_step(self, batch, batch_idx):
        X, label = batch
        loss = self.loss(self.model(X),label)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, label = batch
        loss = self.loss(self.model(X),label)
        self.log("val_loss", loss.item(), prog_bar=True)

    def forward(self, data):
        return self.model.forward(data)
    
    def configure_optimizers(self):
        return SGD(self.model.rot.parameters() , lr=self.learning_rate)

        

