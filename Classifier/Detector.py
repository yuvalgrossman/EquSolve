import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image

# project classes:
from Classifier.HASYDataset import HASYDataset
from Classifier.Net import Net


class Detector():
    def __init__(self, config):
        print('init detector')
        self.config = config
        self.device = self.get_device()
        print(self.device)

        #load network:
        self.theClassifier = self.load_network(config['model_path']).eval().to(self.device)

        self.transform = transforms.Compose([transforms.Resize([28, 28]),
                                             transforms.ToTensor(),
                                             transforms.Normalize(0.5, 0.5),
                                            ])

        # self.class2symbol_mapper = pd.read_csv(config['symbol_list_path']).set_index('symbol_id')['latex']

        # #create dir to save train results:
        # theTime = "{date:%Y-%m-%d_%H-%M-%S}".format(date=datetime.datetime.now())
        # self.Train_Results_Dir = 'DetectionResults/Detection_Results_' + theTime
        # os.mkdir(self.Train_Results_Dir)

    def Detect(self, imgs):

        imgs = torch.stack([self.transform(Image.fromarray(img)) for img in imgs]).type(torch.FloatTensor).to(self.device)

        with torch.no_grad():
            outputs = self.theClassifier(imgs)
            _, predicted = torch.max(outputs.data, 1)

        return [self.class2symbol_mapper[p.item()] for p in predicted]


    def generate_measures_plots(self):
        for key, value in self.tracking_measures.items():
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(value)
            ax.set_title(key)
            plt.grid()
            fn = os.path.join(self.Train_Results_Dir,'{}.png'.format(key))
            plt.savefig(fn)
            plt.close()

    def get_device(self):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    def load_network(self, model_path):
        saved_dict = torch.load(model_path)

        self.class2symbol_mapper = saved_dict['class2sym_mapper']

        net = Net(out_ch = len(self.class2symbol_mapper))

        net.load_state_dict(saved_dict['state_dict'])

        return net
        # saved_dict['train_measures'] = self.tracking_measures
        # saved_dict['config'] = self.config

#     device = get_device()
# train(net, trainloader)
# # test(net, testloader)