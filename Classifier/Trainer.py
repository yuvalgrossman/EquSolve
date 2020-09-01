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
import webbrowser

# project classes:
from Classifier.HASYDataset import HASYDataset
from Classifier.Net import Net


class Trainer():
    def __init__(self, config):
        self.config = config
        self.device = self.get_device()

        #create dir to save train results:
        theTime = "{date:%Y-%m-%d_%H-%M-%S}".format(date=datetime.datetime.now())
        self.Train_Results_Dir = 'TrainResults/Train_Results_' + theTime
        os.mkdir(self.Train_Results_Dir)

        #create and open a webpage monitor: (we just replace one line in the html file to update the folder)
        with open("TrainResults/monitor_base.html") as fin, open("TrainResults/monitor.html", 'w') as fout:
            for line in fin:
                lineout = line
                if 'var results_folder' in line:
                    lineout = 'var results_folder = "Train_Results_{}/"'.format(theTime)
                fout.write(lineout)

        webbrowser.open("TrainResults/monitor.html")

    def train_classifier_HASY(self):

        # DATA LOADER:
        transform = transforms.Compose([transforms.Resize([28, 28]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5),
                                        ])

        dataset = HASYDataset(self.config, transforms=transform)

        test_size = int(self.config['test_fraction'] * len(dataset))
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        trainloader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)

        testloader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=True)

        # TRAINING CONFIGURATIONS:
        net = Net(out_ch = len(dataset.class2sym_mapper)).to(self.device)
        print(net)

        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = optim.SGD(net.parameters(), lr=self.config['lr'], momentum=self.config['momentum'])

        #define tracking measures:
        self.init_tracking_measures()

        # TRAINING:
        print('Start Training on {}'.format(self.device))

        for epochNum in range(self.config['train_epochs']):  # no. of epochs

            net = self.train_epoch(net, trainloader, epochNum)

            self.test_epoch(net, testloader, epochNum)

            self.generate_measures_plots() # update figures after each epoch to observe during training

        self.save_network(net)

        print('Done Training {} epochs'.format(epochNum+1))

        self.generate_measures_plots()

    def init_tracking_measures(self):
        self.tracking_measures = {}
        self.tracking_measures['batch_train_loss'] = []
        self.tracking_measures['batch_train_acc'] = []
        self.tracking_measures['epoch_train_loss'] = []
        self.tracking_measures['epoch_train_acc'] = []
        self.tracking_measures['epoch_test_loss'] = []
        self.tracking_measures['epoch_test_acc'] = []

    def generate_measures_plots(self):
        for key, value in self.tracking_measures.items():
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(value)
            ax.set_title(key)
            plt.grid()
            fn = os.path.join(self.Train_Results_Dir,'{}.png'.format(key))
            plt.savefig(fn)
            plt.close()

    def train_epoch(self, net, trainloader, epoch):

        epoch_loss = 0
        epoch_acc = 0
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

            batch_loss = loss.item()
            epoch_loss += batch_loss
            self.tracking_measures['batch_train_loss'].append(batch_loss)

            _, predicted = torch.max(outputs.data, 1)
            batch_acc = (predicted == labels).sum().item()/len(predicted)
            epoch_acc += batch_acc
            self.tracking_measures['batch_train_acc'].append(batch_acc)

        epoch_loss /= len(trainloader)
        epoch_acc /= len(trainloader)
        print('Train Epoch {} loss: {:.3f} acc: {:.3f}'.format(epoch + 1, epoch_loss, epoch_acc))
        self.tracking_measures['epoch_train_loss'].append(epoch_loss)
        self.tracking_measures['epoch_train_acc'].append(epoch_acc)

        return net


    def test_epoch(self, net, testloader, epoch):
        correct = 0
        total = 0
        epoch_loss = 0
        epoch_acc = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
                # print(inputs.shape, labels.shape)
                outputs = net(inputs)
                loss = self.criterion(outputs, labels)

                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()

                batch_loss = loss.item()
                epoch_loss += batch_loss

                _, predicted = torch.max(outputs.data, 1)
                batch_acc = (predicted == labels).sum().item()/len(predicted)
                epoch_acc += batch_acc

        # print('Accuracy of the network on test images: %0.3f %%' % (
        #         100 * correct / total))
        epoch_loss /= len(testloader)
        epoch_acc /= len(testloader)
        print('Test Epoch {} loss: {:.3f} acc: {:.3f}'.format(epoch + 1, epoch_loss, epoch_acc))
        self.tracking_measures['epoch_test_loss'].append(epoch_loss)
        self.tracking_measures['epoch_test_acc'].append(epoch_acc)

    def get_device(self):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    def save_network(self, net):
        fn = os.path.join(self.Train_Results_Dir, 'HASY_simpleclassifier.pth')
        saved_dict = net.state_dict()
        # add custom data to the saved file:
        saved_dict['train_measures'] = self.tracking_measures
        saved_dict['config'] = self.config
        torch.save(net.state_dict(), fn)

#     device = get_device()
# train(net, trainloader)
# # test(net, testloader)