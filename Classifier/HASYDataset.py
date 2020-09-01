from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Utils.mapper import mapper
import tarfile
import pandas as pd
import os

class HASYDataset(Dataset):
    def __init__(self, config, transforms=None):
        self.config = config
        self.prepare_hasy_dataset()
        self.transforms = transforms

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        y = self.data.loc[idx, 'symbol_id']
        img = self.read_image(idx)
        X = self.transforms(img)   # apply transforms: resize-> tensor-> normalize
        X_reshape = X[0].unsqueeze(-1).transpose(2,0) # reshape to [1,28,28]
        return (X_reshape, y)

    def read_image(self, idx):
        img_path = self.config['data_path'] + self.data.loc[idx, 'path']
        return Image.open(img_path) # load image in PIL format


    def plotitem(self, idx):
        y = self.data.loc[idx, 'latex']
        X = plt.imread(self.config['data_path'] + self.data.loc[idx, 'path'])[:, :, 0]

        plt.imshow(X)
        plt.title(y)
        print('img size {}'.format(X.shape))

    def prepare_hasy_dataset(self):
        if not os.path.exists(self.config['dataset_path']):
            os.system('wget -P DataSets/HASY https://zenodo.org/record/259444/files/HASYv2.tar.bz2?download=1')
            my_tar = tarfile.open('HASYv2.tar.bz2?download=1')
            my_tar.extractall()  # specify which folder to extract to
            my_tar.close()
        full_df = pd.read_csv(self.config['dataset_path'])

        sym_list = None if 'sym_list' not in self.config.keys() else self.config['sym_list']

        self.data = mapper(full_df, sym_list)

        self.class2sym_mapper = np.array(self.data.groupby('symbol_id').first()['latex'])

