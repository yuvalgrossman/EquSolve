import tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from torchvision.transforms import transforms
from torch.utils.data import DataLoader


from Classifier.HASYDataLoader import ExampleDataset
from Classifier.SimpleClassifier import SimpleClassifier


class Trainer():
    def __init__(self, config):
        self.config = config
        self.device = self.get_device()

    def train_classifier_HASY(self):
        # DATA LOADER:

        # if 'hasy-data-labels.csv'

        train_df = pd.read_csv(self.config['data_path'] + self.config['train_data_path'])
        test_df = pd.read_csv(self.config['data_path'] + self.config['test_data_path'])

        train_data = ExampleDataset(self.config, train_df, transforms=transforms.ToTensor())
        trainloader = DataLoader(train_data, batch_size=self.config['batch_size'], shuffle=True)

        test_data = ExampleDataset(self.config, test_df, transforms=transforms.ToTensor())
        testloader = DataLoader(train_data, batch_size=self.config['batch_size'], shuffle=True)

        # TRAINING CONFIGURATIONS:
        net = SimpleClassifier().to(self.device)
        print(net)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        net = self.train(net, trainloader)

        self.save_network(net)

        self.test(net, testloader)


    def train(self, net, trainloader):
        print('Start Training on {}'.format(self.device))

        for epoch in range(self.config['train_epochs']):  # no. of epochs
            running_loss = 0
            for data in tqdm(trainloader):
                # data pixels and labels to GPU if available
                inputs, labels = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
                # print(inputs.shape, labels.shape)
                # set the parameter gradients to zero
                self.optimizer.zero_grad()
                outputs = net(inputs)
                # print(outputs.shape, labels.shape)
                loss = self.criterion(outputs, labels)
                # propagate the loss backward
                loss.backward()
                # update the gradients
                self.optimizer.step()

                running_loss += loss.item()
            print('[Epoch %d] loss: %.3f' %
                  (epoch + 1, running_loss / len(trainloader)))

        print('Done Training')
        return net


    def test(self, net, testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
                # print(inputs.shape, labels.shape)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on test images: %0.3f %%' % (
                100 * correct / total))

    def get_device(self):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    def save_network(self, net):
        PATH = 'Weights/HASY_simpleclassifier.pth'
        torch.save(net.state_dict(), PATH)

#     device = get_device()
# train(net, trainloader)
# # test(net, testloader)