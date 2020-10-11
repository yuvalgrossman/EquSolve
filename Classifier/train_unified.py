from Classifier.Trainer import Trainer
from torchvision.transforms import transforms
import time
inner_path = '/home/yuval/Projects/'
# from trains import Task
# task = Task.init(project_name="EquSolve", task_name="train_hasy")

config = {}
# dataset configurations:
config['DB'] = 'Unified'
config['inner_path'] = inner_path
config['data_path'] = inner_path + 'EquSolve/DataSets/'
config['weights_path'] = inner_path + 'EquSolve/Classifier/weights/'
config['dataset_path'] = inner_path + 'EquSolve/DataSets/HASY/hasy-data-labels.csv'
config['sym_list'] = ['=', '+', '-', 'x']
config['HASY_train_split'] = 0.9

#training configurations:
# config['train_type'] = 'transfer_from_MNIST'
# config['train_type'] = 'continue_HASY'
config['train_type'] = 'new'
config['batch_size'] = 32
config['train_epochs'] = 20
config['lr'] = 0.001
config['momentum'] = 0
config['sampling_evenly'] = True
config['augmentation'] = True

#debugging conf:
config['open_browser'] = True

transform = [ transforms.Resize([28, 28]),
              transforms.ToTensor(),
              transforms.Normalize(0.5, 0.5),
              # transforms.Lambda(lambda x: 1-x) #make the figures white and the background black
            ]

theTrainer = Trainer(config, transform)

tic = time.time()
theTrainer.train()
print('Proccess took {:.2f} m.'.format((time.time() - tic)/60))