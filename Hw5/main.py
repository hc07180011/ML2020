import os, sys, cv2
import numpy as np

from keras.models import load_model
from lib.saliency import saliency
from lib.filter import conv_filter
from lib.plot_lime import plotLime
from lib.deep_dream import process_dream

def readfile(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x

model = load_model('model.h5')

workspace_dir = sys.argv[1]
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
from keras.utils.np_utils import *
train_y = to_categorical(train_y, 11)
train_x = train_x.astype(float) / 255.0

print('Saliency map...')
saliency(sys.argv[2], model, train_x, train_y)
print('Filter...')
conv_filter(sys.argv[2], model, train_x, train_y)
print('Lime...')
plotLime(sys.argv[2], model, train_x, train_y)
print('Deep dream...')
process_dream(sys.argv[2], sys.argv[1])
