import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Classifier.HASYDataset import HASYDataset

# the "=" sign is missing from the dataset. we synthesize it using the sign "-":

def make_equal_sign(mode='same'):
    if mode == 'same':
        idx = random.choice(idxs)
        X = np.array(train_data.read_image(idx))[:,:,0]
        X = 1 - (1 - np.roll(X, 5, axis=0) + (1 - np.roll(X, -5, axis=0)))
    elif mode == 'diff':
        idx1 = random.choice(idxs)
        idx2 = random.choice(idxs)
        X1 = np.array(train_data.read_image(idx1))[:,:,0]
        X2 = np.array(train_data.read_image(idx2))[:,:,0]
        X = 1 - (1 - np.roll(X1, 5, axis=0) + (1 - np.roll(X2, -5, axis=0)))

    return X
    # plt.imshow(X)
    # plt.colorbar()


hasy = pd.read_csv('/home/yuval/Projects/EquSolve/DataSets/HASY/hasy-data-labels.csv')
idxs = hasy.symbol_id[hasy.latex=="-"].index
print(len(idxs))
config = {}
config['data_path'] = '/home/yuval/Projects/EquSolve/DataSets/HASY/'
config['dataset_path'] = '/home/yuval/Projects/EquSolve/DataSets/HASY/hasy-data-labels.csv'
train_data = HASYDataset(config)
train_data.plotitem(idxs[10])
# f, ax = plt.subplots(5, 4, sharex=True, sharey=True)
# ax = ax.reshape(-1)
# for i in range(20):
#     x = make_equal_sign('diff')
#     ax[i].imshow(x, cmap='gray')

# produce and save to dataset:

# df = pd.DataFrame(columns=hasy.columns)
for i in range(200):
    x = make_equal_sign('diff')
    img_path = 'hasy-data/v2-{}.png'.format(len(hasy)+i)
    plt.imsave(config['data_path'] + img_path, x, cmap='gray')
    hasy = hasy.append({'path': img_path, 'symbol_id': 1401, 'latex': '=', 'user_id': 99999}, ignore_index=True)


hasy.to_csv('/home/yuval/Projects/EquSolve/DataSets/HASY/hasy-data-labels.csv')

pass