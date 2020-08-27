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

# project classes:
from Classifier.HASYDataLoader import ExampleDataset
from Classifier.SimpleClassifier import SimpleClassifier


class Detector():
    def __init__(self, config):
        self.config = config
        self.device = self.get_device()

        #load network:
        theClassifier = SimpleClassifier()
        self.theClassifier = self.load_network(theClassifier, config['model_path']).eval()

        self.transform = transforms.Compose([transforms.ToTensor])

        self.class2symbol_mapper = pd.read_csv('DataSets/HASY/symbols.csv').set_index('symbol_id')['latex']

        #create dir to save train results:
        theTime = "{date:%Y-%m-%d_%H-%M-%S}".format(date=datetime.datetime.now())
        self.Train_Results_Dir = 'DetectionResults/Detection_Results_' + theTime
        os.mkdir(self.Train_Results_Dir)

    def Detect(self, imgs):

        imgs = torch.stack([self.transform(img) for img in imgs]).to(self.device)

        with torch.no_grad():
            outputs = self.theClassifier(imgs)
            _, predicted = torch.max(outputs.data, 1)

        return [self.class2symbol_mapper[p] for p in predicted]


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

    def load_network(self, net, model_path):
        saved_dict = torch.load(model_path)
        net.load_state_dict(saved_dict['state_dict'])
        return net
        # saved_dict['train_measures'] = self.tracking_measures
        # saved_dict['config'] = self.config

#     device = get_device()
# train(net, trainloader)
# # test(net, testloader)