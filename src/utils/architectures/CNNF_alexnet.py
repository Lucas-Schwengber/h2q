import torch.nn as nn
from torchvision import models

# This implementation is analogous to the one in
# https://github.com/thuml/MMHH/blob/master/src/mmhh_network.py

class Model(nn.Module):

    NAME = "CNNF_alexnet"

    def __init__(self, number_of_bits):
        super().__init__()
        original_model = models.alexnet(weights='IMAGENET1K_V1')
        self.number_of_bits = number_of_bits
        self.features = original_model.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), original_model.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)
        hash_layer = nn.Linear(original_model.classifier[6].in_features, self.number_of_bits)
        hash_layer.weight.data.normal_(0, 0.01)
        hash_layer.bias.data.fill_(0.0)
        self.hash_layer = hash_layer

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return self.hash_layer(x)