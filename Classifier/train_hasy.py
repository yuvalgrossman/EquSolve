from Classifier.Trainer import Trainer
from torchvision.transforms import transforms
import time
inner_path = '/home/yuval/Projects/'


config = {}
# dataset configurations:
config['state'] = 'HASY'
config['inner_path'] = inner_path
config['data_path'] = inner_path + 'EquSolve/DataSets/HASY/'
config['weights_path'] = inner_path + 'EquSolve/Classifier/weights/'
config['dataset_path'] = inner_path + 'EquSolve/DataSets/HASY/hasy-data-labels.csv'
config['sym_list'] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '+', '-', '\\cdot']
config['HASY_train_split'] = 0.9

#training configurations:
config['batch_size'] = 16
config['train_epochs'] = 15
config['lr'] = 0.001
config['momentum'] = 0.9

transform = transforms.Compose([transforms.Resize([28,28]),
                      transforms.ToTensor(),
                      transforms.Normalize(0.5,0.5),
                      ])

theTrainer = Trainer(config, transform)

tic = time.time()
theTrainer.train()
print('Proccess took {:.2f} m.'.format((time.time() - tic)/60))