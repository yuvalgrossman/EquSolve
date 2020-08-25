from Classifier.Trainer import Trainer
import tarfile
import pandas as pd
import os

def download_dataset():
    # os.system('wget -P DataSets/HASY https://zenodo.org/record/259444/files/HASYv2.tar.bz2?download=1')
    # my_tar = tarfile.open('HASYv2.tar.bz2?download=1')
    # my_tar.extractall()  # specify which folder to extract to
    # my_tar.close()
    meta_data = pd.read_csv('/home/yuval/Projects/EquSolve/DataSets/HASY/hasy-data-labels.csv')
    return meta_data


meta_data = download_dataset()
config = {}
config['data_path'] = '/home/yuval/Projects/EquSolve/DataSets/HASY/'
config['train_data_path'] = 'classification-task/fold-1/train.csv'
config['test_data_path']  = 'classification-task/fold-1/test.csv'
config['batch_size'] = 128
config['train_epochs'] = 1

theTrainer = Trainer(config)
theTrainer.train_classifier_HASY()

