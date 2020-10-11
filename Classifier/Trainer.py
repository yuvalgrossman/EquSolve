import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.datasets import MNIST
from tqdm import tqdm
import webbrowser
import time
import pdb

# project classes:
from Classifier.HASYDataset import HASYDataset
from Classifier.UnifiedDataset import UnifiedDataset
from Classifier.Net import Net



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
                    lineout = 'var results_folder = "Train_Results_{}/"'.format(theTime)
                fout.write(lineout)

        if config['open_browser']:
            webbrowser.open("Classifier/TrainResults/monitor.html")

    def train(self):
        # dataset should come as a tuple of (train_dataset,test_dataset)
        train_data, test_data = self.get_dataset(self.config, self.transform)

        if self.config['sampling_evenly']:
            if self.config['DB'] == 'HASY':
                nClasses = train_data.data.symbol_id.value_counts().sort_index().tolist() # number of labels in each class
            if self.config['DB'] == 'Unified':
                mapper = train_data.hasy.data.symbol_id.value_counts().to_dict()
                nClasses = [6000]*len(train_data.mnist) + [mapper[x] for x in train_data.hasy.data.symbol_id.values]
            sample_weights = torch.tensor([1/n for n in nClasses])
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=self.config['batch_size'])
        else:
            sampler = None

        # move dataset to dataloader
        trainloader = DataLoader(train_data, batch_size=self.config['batch_size'], sampler=sampler)
        testloader = DataLoader(test_data, batch_size=self.config['batch_size'])
        
        # TRAINING CONFIGURATIONS:

        # define tracking measures:
        self.init_tracking_measures()

        weights_save_path = self.config['weights_path'] + self.config['DB'] + '_weights.pth'

        # apply network changes according to training state
        if self.config['DB'] == 'HASY' and self.config['train_type'] == 'transfer_from_MNIST': # if transfer to HASY
            weights_load_path = self.config['weights_path'] + 'MNIST_weights.pth'
            former_model = torch.load(weights_load_path) # load MNIST weights

            net = Net(out_ch=len(former_model['config']['sym_list'])).to(self.device) 
            net.load_state_dict(former_model['state_dict']) # load MNIST weights
            net.fc3 = nn.Linear(84, len(self.config['sym_list'])).to(self.device)   # change model's last layer
            print(net)
            
        elif self.config['DB'] == 'HASY' and self.config['train_type'] == 'continue_HASY': # continue from former train
            weights_load_path = self.config['weights_path'] + 'HASY_weights.pth'
            former_model = torch.load(weights_load_path) # load former model

            net = Net(out_ch=len(self.config['sym_list'])).to(self.device)
            print(net)
            net.load_state_dict(former_model['state_dict']) # load former weights

        else: #new train
            net = Net(out_ch=len(self.config['sym_list'])).to(self.device)
            print(net)

        # loss
        self.criterion = nn.CrossEntropyLoss()

        # optimizer
        self.optimizer = optim.SGD(net.parameters(), lr=self.config['lr'], momentum=self.config['momentum'])


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
            fig, ax = plt.subplots(figsize=(8, 4))
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
        
        for bNum, data in enumerate(trainloader):
            # data pixels and labels to GPU if available
            inputs, labels = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
            if epoch>=0 and bNum==0:
                utils.save_image(inputs, self.Train_Results_Dir+'/train_example_batch.png')

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
        tp = {x:0 for x in self.config['sym_list']}
        tpfp = tp.copy()
        
        net.eval()
        with torch.no_grad():
            for bNum, data in enumerate(testloader):
                inputs, labels = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
                if epoch >= 0 and bNum == 0:
                    utils.save_image(inputs, self.Train_Results_Dir + '/test_example_batch.png')

                outputs = net(inputs)
                loss = self.criterion(outputs, labels)

                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()

                batch_loss = loss.item()
                epoch_loss += batch_loss

                _, predicted = torch.max(outputs.data, 1)
                batch_acc = (predicted == labels).sum().item()/len(predicted)
                epoch_acc += batch_acc
                
              
                for label in labels.unique().tolist():
                  l = labels==label
                  tp[self.config['sym_list'][label]] += sum(predicted[l] == label)
                  tpfp[self.config['sym_list'][label]] += sum(l)
                  
        # print('Accuracy of the network on test images: %0.3f %%' % (
        #         100 * correct / total))
        epoch_loss /= len(testloader)
        epoch_acc /= len(testloader)
        print('Test Epoch {} loss: {:.3f} acc: {:.3f}'.format(epoch + 1, epoch_loss, epoch_acc))
        self.tracking_measures['epoch_test_loss'].append(epoch_loss)
        self.tracking_measures['epoch_test_acc'].append(epoch_acc)
        
        
        for label in self.config['sym_list']:
          class_acc = torch.true_divide(100*tp[label],tpfp[label])
          print('Test acc for {} = {:.3f}% ({}/{})'.format(label,class_acc,tp[label],tpfp[label]))

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

        fn = os.path.join(self.Train_Results_Dir, self.config['DB'] +'.pth')
        torch.save(saved_dict, fn)
        print('save model in ' + fn)

        fn = weights_save_path
        torch.save(saved_dict, fn)
        print('save model in ' + fn)

    def get_dataset(self, config, transform):
      test_transform = transforms.Compose(transform)
      if config['augmentation']:  # train dataset, includes augmentation
        train_transform = transforms.Compose([transforms.RandomRotation(30, fill=255, expand=True)] + transform)
      else:
        train_transform = transforms.Compose(transform)


      if config['DB'] == 'MNIST':
        train_dataset = MNIST(config['data_path'], train=True, download=True, transform=train_transform)
        test_dataset = MNIST(config['data_path'], train=False, download=True, transform=test_transform)
        self.config['sym_list'] = list(range(10))

      if config['DB'] == 'HASY':
        train_dataset = HASYDataset(config, download=True, train=True, transform=train_transform)
        test_dataset = HASYDataset(config, download=True, train=False, transform=test_transform)

      if config['DB'] == 'Unified':
        mnist_train_dataset = MNIST(config['data_path'], train=True, download=True, transform=train_transform)
        mnist_test_dataset = MNIST(config['data_path'], train=False, download=True, transform=test_transform)

        hasy_train_dataset = HASYDataset(config, download=True, train=True, transform=train_transform)
        hasy_test_dataset = HASYDataset(config, download=True, train=False, transform=test_transform)

        train_dataset = UnifiedDataset(mnist_train_dataset, hasy_train_dataset)
        test_dataset = UnifiedDataset(mnist_test_dataset, hasy_test_dataset)
        self.config['sym_list'] = [str(x) for x in range(10)] + self.config['sym_list']

      return (train_dataset,test_dataset)
