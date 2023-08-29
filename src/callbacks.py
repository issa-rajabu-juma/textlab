import matplotlib.pyplot as plt
from keras.callbacks import BaseLogger, Callback
import json
import os
import numpy as np


class EpochCheckpoint(Callback):
    def __init__(self, output_path, every=5, start_at=0):
        super(EpochCheckpoint, self).__init__()
        self.output_path = output_path
        self.every = every
        self.init_epoch = start_at

    def on_epoch_end(self, epoch, logs=None):
        if (self.init_epoch + 1) % self.every == 0:
            p = os.path.sep.join([self.output_path, 'epoch_{}.hdf5'.format(self.init_epoch + 1)])
            self.model.save(p, overwrite=True)

        self.init_epoch += 1


class TrainMonitor(BaseLogger):
    def __init__(self, figure_path, json_path=None, start_at=0):
        super(TrainMonitor, self).__init__()
        self.figure_path = figure_path
        self.json_path = json_path
        self.start_at = start_at

    def on_train_begin(self, logs={}):
        self.H = {}

        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())

                if self.start_at > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_at]

    def on_epoch_end(self, epoch, logs={}):
        for k, v in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        if self.json_path is not None:
            f = open(self.json_path, 'w')
            f.write(json.dumps(self.H))
            f.close()

        if len(self.H['loss']) > 1:
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="train_acc")
            plt.plot(N, self.H["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig(self.figure_path)
            plt.close()
