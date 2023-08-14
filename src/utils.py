from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from imblearn.over_sampling import RandomOverSampler

import os
import numpy as np


def visualize_metric(history, title, metric):
    history = history.history
    epochs = len(history['accuracy'])
    figure(figsize=(8, 6), dpi=80)

    if metric == 'accuracy':
        plt.plot(range(epochs), history['accuracy'], label='Train Accuracy')
        plt.plot(range(epochs), history['val_accuracy'], label='Validation Accuracy')
        plt.ylabel('Accuracy')

    elif metric == 'loss':
        plt.plot(range(EPOCHS), history['loss'], label='Train Loss')
        plt.plot(range(EPOCHS), history['val_loss'], label='Validation loss')
        plt.ylabel('Loss')

    plt.xlabel('Epochs')
    plt.title(title)
    plt.legend()
    plt.show()


def inspect_dataset(dataset):
    for x, y in dataset:
        print('sample shape: {}'.format(x.shape))
        print('target shape: {}'.format(y.shape))
        print('sample type: {}'.format(x.dtype))
        print('target type: {}'.format(y.dtype))
        print('sample: {}'.format(x[0]))
        print('target: {}'.format(y[0]))
        break


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)

    return lowercase


def count_labels(label, labels):
    counter = 0

    for l in labels:
        if l == label:
            counter += 1
        else:
            pass

    return counter


def roser(dataset, batch_size=None):
    X = []
    Y = []
    ROS = RandomOverSampler(random_state=0)

    if batch_size is None:
        for x, y in dataset:
            X.append(x)
            Y.append(y)
        X, Y = ROS.fit_resample(X, Y)
        dataset = mkds(X, Y)

    else:
        for x, y in dataset.unbatch():
            X.append(x)
            Y.append(y)
        X, Y = ROS.fit_resample(X, Y)
        dataset = mkds(X, Y, batch_size)

    return dataset


def mkds(x, y, batch_size):
    if batch_size is None:
        return tf.data.Dataset.from_tensor_slices((x, y))
    else:
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)


def get_split_files(base_dir, split):
    file_list = []
    for dirName, subdirList, fileList in os.walk(base_dir):
        for file in fileList:
            if file == split:
                file_list.append(os.path.join(dirName, file))

    return file_list


def load_data(file_list):
    # def load_data(file_list):
    data_in_file = []
    samples = []
    targets = []
    count = 0

    # read in-file data
    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data_in_file.append(line)

    # deduce sample and target
    for i, line in enumerate(data_in_file):
        data = line.split(' ')
        if len(data) == 2:
            samples.append(data[0])
            targets.append(data[1].strip('\n'))
            count += 1
        else:
            continue

    # encode labels
    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(targets)

    # convert to numpy array
    samples = np.array(samples)
    targets = np.array(targets)

    return samples, targets
