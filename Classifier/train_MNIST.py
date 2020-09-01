from Classifier.Trainer import Trainer
from torchvision.transforms import transforms
import time
inner_path = '/home/yuval/Projects/'


config = {}
# dataset configurations:
config['inner_path'] = inner_path
config['data_path'] = inner_path + 'EquSolve/DataSets/'
config['weights_path'] = inner_path + 'EquSolve/Classifier/weights/'

#training configurations:
config['batch_size'] = 128
config['train_epochs'] = 5
config['lr'] = 0.01
config['momentum'] = 0.9
config['state'] = 'MNIST'


transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(0.5,0.5),
                              ])

theTrainer = Trainer(config, transform)

tic = time.time()
theTrainer.train()
print('Proccess took {:.2f} m.'.format((time.time() - tic)/60))