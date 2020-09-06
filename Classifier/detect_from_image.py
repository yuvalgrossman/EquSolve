import cv2
import numpy as np
import matplotlib.pyplot as plt
from Utils.symbols_seperation import *
from Classifier.Detector import Detector
inner_path = '/home/yuval/Projects/'

data_path = inner_path + 'EquSolve/DataSets/hand_written_eqs/'
fn = data_path + 'single_eq5.png'
img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

cc = find_cc(img, 140)
cc = unite_symbols_with_gap(cc)
plot_detections(img, cc)

c = crop_resize((img < 140).astype('uint8'), cc)
fig, ax = plt.subplots(1, len(c), sharey=True)
[ax[i].imshow(c[i]) for i in range(len(c))]
# plt.show()

config = {}
config['model_path'] = inner_path + 'EquSolve/Classifier/weights/HASY_weights.pth'

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

# print(detect_symbols_from_image(inner_path + 'EquSolve/DataSets/hand_written_eqs/single_eq2.jpg'))