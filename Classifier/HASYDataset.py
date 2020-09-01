from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class HASYDataset(Dataset):
    def __init__(self, config, csv_df,transforms):
        self.config = config
        self.data = csv_df
        self.transforms = transforms

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        y = self.data.loc[idx, 'symbol_id']
        img_path = self.config['data_path'] + self.data.loc[idx, 'path']
        img = Image.open(img_path) # load image in PIL format
        X = self.transforms(img)   # apply transforms: resize-> tensor-> normalize
        X_reshape = X[0].unsqueeze(-1).transpose(2,0) # reshape to [1,28,28]
        return (X_reshape, y)

    def plotitem(self, idx):
        y = self.data.loc[idx, 'latex']
        X = plt.imread(self.config['data_path'] + self.data.loc[idx, 'path'])[:, :, 0]

        plt.imshow(X)
        plt.title(y)
        print('img size {}'.format(X.shape))
