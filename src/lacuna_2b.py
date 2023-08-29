import sys
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from keras.callbacks import LearningRateScheduler, EarlyStopping
from callbacks import EpochCheckpoint
import config
from models import build_lacuna, build, build_3bt
import utils
import os
import tensorflow.keras.backend as K
from callbacks import TrainMonitor
import optimizers as opt
import numpy as np
import keras_tuner
from layers import TransformerEncoder


# extract data files
train_files = utils.get_split_files(config.base_dir, 'train.txt')
test_files = utils.get_split_files(config.base_dir, 'test.txt')
dev_files = utils.get_split_files(config.base_dir, 'dev.txt')

# load data
train_x, train_y = utils.load_data(train_files)
test_x, test_y = utils.load_data(test_files)
dev_x, dev_y = utils.load_data(dev_files)

# get label size
label_size = len(set(train_y))

# create dataset
lac_train_ds = utils.mkds(train_x, train_y, config.BATCHSIZE)
lac_test_ds = utils.mkds(test_x, test_y, config.BATCHSIZE)
lac_dev_ds = utils.mkds(dev_x, dev_y, config.BATCHSIZE)

# data shuffling
lac_train_ds = lac_train_ds.shuffle(config.SEED)
lac_test_ds = lac_test_ds.shuffle(config.SEED)
lac_dev_ds = lac_dev_ds.shuffle(config.SEED)

# data transformation
lac_vectorization = keras.layers.TextVectorization(output_mode='int', max_tokens=config.VOCABSIZE,
                                                   output_sequence_length=1)

# adapt vectorization
lac_text_ds = lac_train_ds.map(lambda x, y: x)
lac_vectorization.adapt(lac_text_ds)

# vectorization
lac_int_train_ds = lac_train_ds.map(lambda x, y: (lac_vectorization(x), y), num_parallel_calls=config.NPCALLS)
lac_int_test_ds = lac_test_ds.map(lambda x, y: (lac_vectorization(x), y), num_parallel_calls=config.NPCALLS)
lac_int_dev_ds = lac_dev_ds.map(lambda x, y: (lac_vectorization(x), y), num_parallel_calls=config.NPCALLS)

# unpack dataset
x_train, y_train = utils.unpack_ds(lac_int_train_ds)
x_test, y_test = utils.unpack_ds(lac_int_test_ds)
x_dev, y_dev = utils.unpack_ds(lac_int_dev_ds)

# compute class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))

# over sampling
# train_dataset = utils.roser(lac_int_train_ds, config.BATCHSIZE)
# dev_dataset = utils.roser(lac_int_dev_ds, config.BATCHSIZE)

# utils.inspect_dataset(train_dataset)
# utils.inspect_dataset(dev_dataset)

# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# y_train_scaled = scaler.fit_transform(y_train)
# x_dev_scaled = scaler.fit_transform(x_dev)
# y_dev_scaled = scaler.fit_transform(y_dev)


# print(x_train_scaled[:100])
# print(x_dev_scaled[:100])
#
# print(type(x_train_scaled))
# print(type(x_dev_scaled))
# print()
# sys.exit()
#
# # model initialization
# hp = keras_tuner.HyperParameters()
# lacuna_model = build_lacuna(hp)
#
# # prepare search space
# tuner = keras_tuner.RandomSearch(hypermodel=build_lacuna,
#                                  objective='val_loss',
#                                  max_trials=3,
#                                  executions_per_trial=3,
#                                  overwrite=True,
#                                  directory=config.TUNERDIR,
#                                  project_name='lacuna')
# tuner.search_space_summary()
#
# # start searching
# tuner.search(x_train, y_train, epochs=5, validation_data=(x_dev, y_dev), batch_size=config.BATCHSIZE)
#
#
# # get best hyperparameters
# best_hps = tuner.get_best_hyperparameters(5)
# print(best_hps[0])
#
# print()
# tuner.results_summary()

# # build a model and train
# model = build_lacuna(best_hps[0])
#
# # print summary and plot a model
# model.summary()
# plot_model(model, config.lacuna_2b_architecture_path, show_shapes=True)




# training

if config.MODEL is None:
    # model = build_lacuna(best_hps[0])
    model = build_3bt(config.VOCABSIZE, config.EMBEDDIM, config.DENSEDIM, config.OUTPUTDIM)
    # model = build()
    model.summary()
    plot_model(model, config.ARCHITECTUREPATH, show_shapes=True)

else:
    # load checkpoint from disk
    model = keras.models.load_model(config.MODEL, custom_objects={'TransformerEncoder': TransformerEncoder})

    # old learning rate
    old_lr = K.get_value(model.optimizer.lr)
    print("[INFO] old learning rate: {}".format(old_lr))

    # update the learning rate
    model.optimizer.lr.assign(config.NEWLR)
    new_lr = K.get_value(model.optimizer.learning_rate)
    print("[INFO] new learning rate: {}".format(new_lr))


callbacks = [
    keras.callbacks.ModelCheckpoint(config.KERASMODEL, monitor='val_loss', mode='min', save_best_only=True,
                                    verbose=1),
    TrainMonitor(config.PLOTPROGRESS, config.JSONPROGRESS, config.STARTAT),
    EpochCheckpoint(config.OUTPUTPATH, every=config.SAVEEVERY, start_at=config.STARTAT), 
    # LearningRateScheduler(opt.step_decay),
    EarlyStopping(monitor='val_loss', patience=5, verbose=1),
    keras.callbacks.TensorBoard(log_dir=config.LOGDIR, histogram_freq=1)
]

lacuna_2b_model_history = model.fit(x_train, y_train,
                                    validation_data=(x_dev, y_dev),
                                    epochs=config.EPOCHS,
                                    batch_size=config.BATCHSIZE,
                                    callbacks=callbacks,
                                    class_weight=class_weights)
