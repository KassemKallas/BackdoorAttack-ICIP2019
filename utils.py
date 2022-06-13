# example of loading the mnist dataset
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, LearningRateScheduler
import os
from sklearn.model_selection import train_test_split
import tempfile
from parameters import *
from triggers import *

def make_confusion_matrix(cf, savePath, saveName,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.

    savePath:      where to save the plot

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names )==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten( ) /np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels ,group_counts ,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0] ,cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf )==2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1 ,1] / sum(cf[: ,1])
            recall    = cf[1 ,1] / sum(cf[1 ,:])
            f1_score  = 2* precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    plt.savefig(Path(savePath) / (str(saveName) + '.jpg'))


def plot_confusion_matrix(cm, save_path, save_name, categories_digits):
    # ax = plt.subplot()
    # sns.heatmap(cm, annot=True, ax=ax, fmt='g')  # annot=True to annotate cells
    ## labels, title and ticks
    # ax.set_xlabel('Predicted', fontsize=20)
    # ax.xaxis.set_label_position('top')
    # ax.xaxis.set_ticklabels(['0', '1', '2', '3'], fontsize=15)
    # ax.xaxis.tick_top()

    # ax.set_ylabel('True', fontsize=20)
    # ax.yaxis.set_ticklabels(['0', '1', '2', '3'], fontsize=15)
    ##plt.show()
    # plt.savefig(Path(save_path) / ('confusion_matrix' + '.jpg'))
    make_confusion_matrix(cf=cm, savePath=save_path, saveName=save_name,
                          group_names=None,
                          categories=categories_digits,
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=(12, 8),
                          cmap='Blues',
                          title=None)


def stat_from_cm(cf): # statistics from confusion matrix
    accuracy = np.trace(cf) / float(np.sum(cf))

    # if it is a binary confusion matrix, show some more stats
    if len(cf) == 2:
        # Metrics for Binary Confusion Matrices
        precision = cf[1, 1] / sum(cf[:, 1])
        recall = cf[1, 1] / sum(cf[1, :])
        f1_score = 2 * precision * recall / (precision + recall)
    if len(cf) == 2:
        return accuracy, precision, recall, f1_score
    else:
        return accuracy


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def train(params):
    res_dir = params.get('res_dir')
    subdir_name = params.get('subdir_name')
    x_train = params.get('x_train')
    x_val = params.get('x_val')
    x_test = params.get('x_test')
    y_train = params.get('y_train')
    y_val = params.get('y_val')
    y_test = params.get('y_test')
    model = params.get('model')
    attack = params.get('attack')

    x_train_poisoned = np.copy(x_train)
    x_test_poisoned = np.copy(x_test)



    if nb_channels == 1:
        trigger_dim = (x_train.shape[1], x_train.shape[2], 1)
    else:
        trigger_dim = x_train.shape[1:]

    if attack.trigger == 'ramp':
        trigger = ramp_signal(trigger_dim)
    elif attack.trigger == 'biramp':
        trigger = biramp_signal(trigger_dim)
    else:  # when sinusoidal
        trigger = sinusoidal_signal(trigger_dim, attack.sin_frequency)

    indexes_target_class = [idx for idx, val in enumerate(y_train) if y_train[idx] == attack.targetClass]
    nb_images_to_poison = int(len(indexes_target_class) * attack.alfa)
    index_to_poison = [np.random.choice(indexes_target_class, size = nb_images_to_poison, replace=False)][0]

    poisoned_samples = np.uint8(np.add(x_train_poisoned[index_to_poison], attack.delta * trigger))
    poisoned_samples = np.clip(poisoned_samples, 0, 255)
    x_train_poisoned[index_to_poison,] = poisoned_samples

    x_train_poisoned = np.expand_dims(x_train_poisoned, -1)

    x_test_poisoned = np.uint8(np.add(x_test_poisoned, attack.delta_test * trigger))
    x_test_poisoned = np.clip(x_test_poisoned, 0, 255)
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)


    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # BUILD CONVOLUTIONAL NEURAL NETWORKS
    model_name_call_back = os.path.join(subdir_name ,"model_AttackAlfa_" + str(attack.alfa) + "model_AttackDeltatr_" + str(attack.delta) +"_at_{epoch}.h5")

    csv_logger = CSVLogger(os.path.join(subdir_name, 'MNIST_Backdoor_AttackAlfa_' + str(attack.alfa) + "_AttackDeltatr_" + str(attack.delta) + '.csv'))
    model_checkpoint = ModelCheckpoint(model_name_call_back, save_best_only=False)

    callbacks = [csv_logger, model_checkpoint]

    history = model.fit(x_train_poisoned, y_train, validation_data=(x_val, y_val), batch_size=batch_size,
                        callbacks=callbacks, epochs=epochs)

    # CONFUSION MATRIX ON BENING DATA
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=-1)  # np.rint(predictions)

    y_true = np.argmax(y_test, axis=-1)
    cm = confusion_matrix(y_true, y_pred)  # , labels=[1,0])

    stats_benign = stat_from_cm(cm)
    print("Perf test benign")
    print(stats_benign)
    #print(cm)
    plot_confusion_matrix(cm, subdir_name, 'confusion_matrix_model_benign_AttackAlfa_' + str(attack.alfa)+  "_AttackDeltatr_" + str(attack.delta),
                          categories_digits=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    plt.close()


    # CONFUSION MATRIX ON POISONED DATA
    predictions = model.predict(x_test_poisoned)
    y_pred = np.argmax(predictions, axis=-1)  # np.rint(predictions)

    y_true = np.argmax(y_test, axis=-1)

    cm = confusion_matrix(y_true, y_pred)  # , labels=[1,0])

    stats_poisoned = stat_from_cm(cm)
    #print(cm)
    plot_confusion_matrix(cm, subdir_name, 'confusion_matrix_model_adversarial_AttackAlfa_' + str(attack.alfa)+ "_AttackDeltatr_" + str(attack.delta),
                          categories_digits=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    plt.close()


    plt.figure(1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(subdir_name, 'accuracy_plot_benign_model_AttackAlfa_' + str(attack.alfa) + "_AttackDeltatr_" + str(attack.delta) +'.jpg'))
    plt.close()

    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    plt.savefig(os.path.join(subdir_name, 'loss_plot_benign_model_AttackAlfa_' + str(attack.alfa) + "_AttackDeltatr_" + str(attack.delta) +'.jpg'))

    plt.close()

    logdir = tempfile.mkdtemp()
    print('Writing training logs to ' + logdir)

    _, model_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model, model_file, include_optimizer=False)

    return stats_benign, stats_poisoned, model, model_file, x_test_poisoned