import random
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Classifier.HASYDataset import HASYDataset

# the "=" sign is missing from the dataset. we synthesize it using the sign "-":
config = {}
config['data_path'] = '/home/yuval/Projects/EquSolve/DataSets/HASY/'
config['dataset_path'] = '/home/yuval/Projects/EquSolve/DataSets/HASY/hasy-data-labels.csv'
hasy = pd.read_csv(config['dataset_path'])
idxs = hasy.symbol_id[hasy.latex=="-"].index
print(len(idxs))


def read_img(idx):
    img_path = config['data_path'] + hasy.loc[idx, 'path']
    return cv2.imread(img_path, 0)

def make_equal_sign(mode='same'):
    if mode == 'same':
        idx = random.choice(idxs)
        X = np.array(read_img(idx))
        X = (1 - np.roll(X, 5, axis=0) + (1 - np.roll(X, -5, axis=0)))
    elif mode == 'diff':
        idx1 = random.choice(idxs)
        idx2 = random.choice(idxs)
        X1 = np.array(read_img(idx1))
        X2 = np.array(read_img(idx2))
        X = (1 - np.roll(X1, 5, axis=0) + (1 - np.roll(X2, -5, axis=0)))

    return X
    # plt.imshow(X)
    # plt.colorbar()



# train_data = HASYDataset(config, pd.read_csv(config['dataset_path']))
# train_data.plotitem(idxs[10])
# f, ax = plt.subplots(5, 4, sharex=True, sharey=True)
# ax = ax.reshape(-1)
# for i in range(20):
#     x = make_equal_sign('diff')
#     ax[i].imshow(x, cmap='gray')

# produce and save to dataset:

# df = pd.DataFrame(columns=hasy.columns)
l = len(hasy)
for i in range(150):
    x = make_equal_sign('diff')
    img_path = 'hasy-data/v2-{}.png'.format(l+i)
    fn = config['data_path'] + img_path
    plt.imsave(config['data_path'] + img_path, x, cmap='gray')
    print('created ' + fn)
    hasy = hasy.append({'path': img_path, 'symbol_id': 1401, 'latex': '=', 'user_id': 99999}, ignore_index=True)


hasy.to_csv('/home/yuval/Projects/EquSolve/DataSets/HASY/hasy-data-labels.csv')

pass