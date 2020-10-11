from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from Utils.mapper import mapper
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split


class HASYDataset(Dataset):

    def __init__(self, config, train=True, transform=transforms.Compose([]), download=False):
        self.config = config
        self.root = os.path.join(config['data_path'], 'HASY')
        self.transform = transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        meta_data = pd.read_csv(os.path.join(self.root, 'hasy-data-labels.csv'))

        all_df = mapper(meta_data, config['sym_list'])  # slice only needed symbols

        # split dataframe into train test (before creating the dataset, so we can use different transform):
        train_df, test_df = train_test_split(all_df, train_size=config['HASY_train_split'], random_state=42,
                                         shuffle=True)

        if train:
            self.data = train_df.reset_index()
        else:
            self.data = test_df.reset_index()

        print(self.data.latex.value_counts())


    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        y = self.data.loc[idx, 'symbol_id']
        img_path = os.path.join(self.root, self.data.loc[idx, 'path'])
        img = Image.open(img_path) # load image in PIL format
        X = self.transform(img)   # apply transforms: resize-> tensor-> normalize
        X = X[0].unsqueeze(0) # reshape to [1,28,28]
        X = 1-X #make the figures white and the background black
        return (X, y)

    def plotitem(self, idx):
        y = self.data.loc[idx, 'latex']
        X = plt.imread(os.path.join(self.root, self.data.loc[idx, 'path']))[:, :, 0]

        plt.imshow(X)
        plt.title(y)
        print('img size {}'.format(X.shape))

    def download(self):
        if not self._check_exists():  # download data
            import tarfile
            import requests
            url = 'https://zenodo.org/record/259444/files/HASYv2.tar.bz2?download=1'
            out = os.path.join(self.root, 'HASYv2.tar')
            print('Downloading HASY dataset')
            r = requests.get(url)
            with open(out, 'wb') as f:
                f.write(r.content)

            my_tar = tarfile.open(out)
            print('Extracting dataset')
            my_tar.extractall(self.root)  # specify which folder to extract to
            my_tar.close()
            print('Done extracting')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'hasy-data'))
