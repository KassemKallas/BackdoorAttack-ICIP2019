# example of loading the mnist dataset
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, LearningRateScheduler
import os
from sklearn.model_selection import train_test_split
import random

from parameters import *
from utils import *
from triggers import *
from models import *


#create instance of attack arameters defined in parameters.py
attack = Attack()

#create directory results and subdirectories based on attack.alfa
res_dir = 'results'
isExist = os.path.exists(res_dir)
if not isExist:
  os.mkdir(res_dir)

subdir_name = os.path.join(res_dir, str(attack.alfa))
isExist = os.path.exists(subdir_name)
if not isExist:
  os.mkdir(subdir_name)

## load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size= validation_ratio)
x_train = np.expand_dims(x_train, -1)
x_val = np.expand_dims(x_val, -1)
x_test = np.expand_dims(x_test, -1)


#create CNN used for the experiment
model = lenet5(input_shape=input_shape, nb_classes=nb_classes, drop=False)

# pack necessary parameters for attack
args_attack = {'res_dir': res_dir, 'subdir_name': subdir_name, 'x_train': x_train,
               'y_train': y_train, 'x_val': x_val, 'y_val': y_val, 'x_test': x_test, 'y_test': y_test, 'model': model,
               'attack': attack}

# train with attacker
acc_benign, acc_poisoned, model_trained, model_file, x_test_poisoned = train(args_attack)
print("acc_on_benign_data={0},acc_on_poisoned_data={1}".format(acc_benign, acc_poisoned))
