import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import webbrowser
import time
import pdb

# project classes:
from Classifier.HASYDataset import HASYDataset
from Classifier.Net import Net
from Utils.mapper import mapper


class Trainer():
    def __init__(self, config, transform):
        self.config = config
        self.device = self.get_device()
        self.transform = transform

        #create dir to save train results:
        theTime = "{date:%Y-%m-%d_%H-%M-%S}".format(date=datetime.datetime.now())
        self.Train_Results_Dir = 'Classifier/TrainResults/Train_Results_' + theTime
        os.mkdir(self.Train_Results_Dir)

        #create and open a webpage monitor: (we just replace one line in the html file to update the folder)
        with open("Classifier/TrainResults/monitor_base.html") as fin, open("Classifier/TrainResults/monitor.html", 'w') as fout:
            for line in fin:
                lineout = line
                if 'var results_folder' in line:
                    lineout = 'var results_folder = "Classifier/Train_Results_{}/"'.format(theTime)
                fout.write(lineout)

        webbrowser.open("Classifier/TrainResults/monitor.html")

    def train(self):
        # dataset should come as a tuple of (train_dataset,test_dataset)
        dataset = download_dataset(self.config, self.transform)
        train_data = dataset[0]
        test_data  = dataset[1]

        # move dataset to dataloader
        trainloader = DataLoader(train_data, batch_size=self.config['batch_size'], shuffle=True)
        testloader = DataLoader(test_data, batch_size=self.config['batch_size'], shuffle=True)
        
        # TRAINING CONFIGURATIONS:
        net = Net().to(self.device)
        print(net)

        # loss
        self.criterion = nn.CrossEntropyLoss()

        # optimizer
        self.optimizer = optim.SGD(net.parameters(), lr=self.config['lr'], momentum=self.config['momentum'])

        # define tracking measures:
        self.init_tracking_measures()

        # apply network changes according to training state
        if self.config['state'] == 'MNIST': # if training on MNIST
            weights_save_path = self.config['weights_path'] + 'MNIST_weights.pth'

        if self.config['state'] == 'HASY': # if training on HASY
            weights_load_path = self.config['weights_path'] + 'MNIST_weights.pth'
            weights_save_path = self.config['weights_path'] + 'HASY_weights.pth'

            net.load_state_dict(torch.load(weights_load_path)['state_dict']) # load MNIST weights
            net.fc3 = nn.Linear(84, len(config['sym_list'])).to(self.device)   # change model's last layer


        # TRAINING:
        print('Start Training on {}'.format(self.device))

        for epochNum in range(self.config['train_epochs']):  # no. of epochs

            net = self.train_epoch(net, trainloader, epochNum)

            self.test_epoch(net, testloader, epochNum)

            self.generate_measures_plots() # update figures after each epoch to observe during training

        self.save_network(net, weights_save_path)

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
        net.train()
        
        for data in tqdm(trainloader):
            # data pixels and labels to GPU if available
            inputs, labels = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
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

        net.eval()
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
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

    def save_network(self, net, weights_save_path):
        saved_dict = {'state_dict': net.state_dict()}
        # add custom data to the saved file:
        saved_dict['train_measures'] = self.tracking_measures
        saved_dict['config'] = self.config
        # saved_dict['class2sym_mapper'] = class2sym_mapper

        fn = os.path.join(self.Train_Results_Dir,config['state'] +'.pth')
        torch.save(saved_dict, fn)
        print('save model in ' + fn)

        fn = weights_save_path
        torch.save(saved_dict, fn)
        print('save model in ' + fn)

    def download_dataset(config, transform):
      if config['state'] == 'MNIST':
        import torchvision
        train_dataset = torchvision.datasets.MNIST(config['data_path'], train=True, download=True,
                              transform=transform)
        test_dataset = torchvision.datasets.MNIST(config['data_path'], train=False, download=True,
                              transform=transform)                      


      if config['state'] == 'HASY':
        if not os.path.exists(config['data_path'] + 'hasy-data'): # download data  
          import tarfile
          import requests    
          url = 'https://zenodo.org/record/259444/files/HASYv2.tar.bz2?download=1'
          out = config['data_path'] + 'HASYv2.tar'
          print('Downloading HASY dataset')
          r = requests.get(url)
          with open(out, 'wb') as f:
              f.write(r.content)
          
          my_tar = tarfile.open(out)
          print('Extracting dataset')
          my_tar.extractall(config['data_path'])  # specify which folder to extract to
          my_tar.close()
          print('Done extracting')
          
        meta_data = pd.read_csv(config['data_path'] + 'hasy-data-labels.csv')
        # here we concatenate all_df with equal sign df
        all_df = mapper(meta_data,config['sym_list']) # slice only needed symbols
        print(all_df.latex.value_counts())

        dataset = HASYDataset(config,all_df,transform) # read data into dataset
        train_size = int(config['HASY_train_split'] * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                    [train_size, test_size]) # split dataset to train and test

      return (train_dataset,test_dataset)
