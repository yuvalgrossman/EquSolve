from Classifier.Trainer import Trainer
import time
inner_path = '/home/yuval/Projects/'


config = {}
# dataset configurations:
config['inner_path'] = inner_path
config['data_path'] = inner_path + 'EquSolve/DataSets/HASY/'
config['dataset_path'] = inner_path + 'EquSolve/DataSets/HASY/hasy-data-labels.csv'
config['sym_list'] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '\\alpha', '=', '+', '-', '\\pi', 'A', 'X', '\\cdot']
config['test_fraction'] = 0.1

#training configurations:
config['batch_size'] = 128
config['train_epochs'] = 100
config['lr'] = 0.01
config['momentum'] = 0.9

theTrainer = Trainer(config)

tic = time.time()
theTrainer.train_classifier_HASY()
print('Proccess took {:.2f} m.'.format((time.time() - tic))/60)

