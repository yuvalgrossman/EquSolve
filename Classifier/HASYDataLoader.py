from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class ExampleDataset(Dataset):
    def __init__(self, config, csv_df, transforms=None):
        self.config = config
        self.data = csv_df
        self.transforms = transforms

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        y = self.data.loc[idx, 'symbol_id']
        X = np.expand_dims(plt.imread(self.config['data_path'] + self.data.loc[idx, 'path'][6:])[:, :, 0], 2)

        if self.transforms:
            X = self.transforms(X)

        return (X, y)

    def plotitem(self, idx):
        y = self.data.loc[idx, 'latex']
        X = plt.imread(self.config['data_path'] + self.data.loc[idx, 'path'][6:])[:, :, 0]

        plt.imshow(X)
        plt.title(y)
        print('img size {}'.format(X.shape))
        # print((X[:,:,2]==X[:,:,0]).all())
        # print((X[:,:,2]==X[:,:,1]).all())
