import pathlib

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from utils.general_utils import load_features_hashes_and_labels


# Dataset for images
class VectorizedDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        datafold,
        load_matches=2,
        transformations="imagenet"
    ):
        super().__init__()
        self.dataset_dir = pathlib.Path(f"data/processed/{dataset_name}")
        self.datafold = datafold
        self.load_matches = load_matches
        self.rng = np.random.default_rng(42)

        # load entries
        with open(self.dataset_dir / f"{self.datafold}_metadata.txt") as f:
            metadata = f.readlines()
        self.filenames = [m.split(",")[0] for m in metadata]
        self.labels = np.array([[float(v) for v in m.split(",")[1].split(" ")] for m in metadata])
        self.data_len = len(self.filenames)
        self.n_classes = self.labels.shape[1]
        
        # define transforms
        if transformations == "imagenet":
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def get_image(self, i):
        image = Image.open(self.filenames[i])
        image = image.convert('RGB')
        # image = np.moveaxis(np.array(image) / 255.0, -1, 0)
        return self.transform(image) #image.astype(np.float32)

    def get_similar_probability(self):
        return np.mean(1*(self.labels @ self.labels.T > 0))

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return (self.get_image(idx), self.labels[idx])


# Dataset to load pre-processed embeddings
class EmbeddedDataset(Dataset):
    def __init__(
        self,
        model_dir,
        datafold,
        normalize=True,
        reverse_tanh=False
    ):
        super().__init__()
        self.model_dir = pathlib.Path(model_dir)
        self.datafold = datafold
        self.normalize = normalize
        self.reverse_tanh = reverse_tanh

        # load data
        self.features, _, self.labels = load_features_hashes_and_labels(model_dir, datafold)

        if self.reverse_tanh:
            eps=1e-7
            self.features = np.arctanh(self.features*(1-eps))

        if self.normalize:
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
