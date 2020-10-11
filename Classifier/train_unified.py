from Classifier.Trainer import Trainer
from torchvision.transforms import transforms as T
from albumentations.augmentations import transforms as A
from Utils.ImageTransforms import GaussianSmoothing, Brightenn, Contraster
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
config['augmentation'] = [T.RandomRotation(15, fill=255, expand=True),
                          T.RandomHorizontalFlip(),
                          T.RandomVerticalFlip(),
                          Brightenn(0.4),
                          Contraster(10),
                          # GaussianSmoothing([0, 7])
                          ]
# config['augmentation'] = None

#debugging conf:
config['open_browser'] = True

transform = [ T.Resize([28, 28]),
              T.ToTensor(),
              T.Normalize(0.5, 0.5),
              # transforms.Lambda(lambda x: 1-x) #make the figures white and the background black - happens inside dataset
            ]

theTrainer = Trainer(config, transform)

tic = time.time()
theTrainer.train()
print('Proccess took {:.2f} m.'.format((time.time() - tic)/60))