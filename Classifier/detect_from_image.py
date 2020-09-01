import cv2
import numpy as np
import matplotlib.pyplot as plt
from Utils.symbols_seperation import *
from Classifier.Detector import Detector

data_path = '/home/yuval/Projects/EquSolve/DataSets/hand_written_eqs/'
fn = data_path + 'single_eq1.jpg'
img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

cc = find_cc(img, 160)
cc = unite_symbols_with_gap(cc)
plot_detections(img, cc)

c = crop_resize((img < 160).astype('uint8'), cc)
fig, ax = plt.subplots(1, len(c), sharey=True)
[ax[i].imshow(c[i]) for i in range(len(c))]
# plt.show()

config = {}
config['model_path'] = '/home/yuval/Projects/EquSolve/Classifier/TrainResults/Train_Results_2020-09-01_09-30-50/HASY_simpleclassifier.pth'
# config['symbol_list_path'] = '/home/yuval/Projects/EquSolve/DataSets/HASY/symbols.csv'

theClassifier = Detector(config)

detected_symbols = theClassifier.Detect(c)

[ax[i].imshow(c[i]) for i in range(len(c))]
plt.suptitle(detected_symbols)
plt.show()


def detect_symbols_from_image(img_fn):
    img = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
    cc = find_cc(img, 160)
    cc = unite_symbols_with_gap(cc)
    c = crop_resize((img < 160).astype('uint8'), cc)
    return theClassifier.Detect(c)

print(detect_symbols_from_image('/home/yuval/Projects/EquSolve/DataSets/hand_written_eqs/single_eq2.jpg'))