import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

class ImageDatasetProcessing(Dataset):
    def __init__(self, metadata_file_path, transform=None):
        self.metadata_path = metadata_file_path
        self.transform = transform
        # reading img file from file
        with open(self.metadata_path) as f:
            metadata = f.readlines()

        self.img_filename = [m.split(",")[0] for m in metadata]
        self.label = np.array([[int(v) for v in m.split(",")[1].split(" ")] for m in metadata])
 
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor(np.array([self.label[index]]))
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


class DatasetProcessingNUS_WIDE(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


class DatasetProcessingMS_COCO(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)