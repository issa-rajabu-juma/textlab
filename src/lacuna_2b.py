import sys

from tensorflow import keras
from tensorflow.keras.utils import plot_model
from keras.callbacks import LearningRateScheduler
import config
from models import build_2b_lacuna, build_dense_lacuna, build_simple_lacuna
import utils
import os
from callbacks import TrainMonitor
import optimizers as opt
import keras_tuner

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

lac_train_ds = lac_train_ds.shuffle(config.SEED)
lac_test_ds = lac_test_ds.shuffle(config.SEED)
lac_dev_ds = lac_dev_ds.shuffle(config.SEED)

# data transformation
lac_vectorization = keras.layers.TextVectorization(output_mode='int', max_tokens=config.VOCABSIZE,
                                                   output_sequence_length=1)
lac_text_ds = lac_train_ds.map(lambda x, y: x)
lac_vectorization.adapt(lac_text_ds)

lac_int_train_ds = lac_train_ds.map(lambda x, y: (lac_vectorization(x), y), num_parallel_calls=config.NPCALLS)
lac_int_test_ds = lac_test_ds.map(lambda x, y: (lac_vectorization(x), y), num_parallel_calls=config.NPCALLS)
lac_int_dev_ds = lac_dev_ds.map(lambda x, y: (lac_vectorization(x), y), num_parallel_calls=config.NPCALLS)

train_dataset = utils.roser(lac_int_train_ds, config.BATCHSIZE)
dev_dataset = utils.roser(lac_int_dev_ds, config.BATCHSIZE)

utils.inspect_dataset(train_dataset)
utils.inspect_dataset(dev_dataset)
print()


# tuner code
#
# hp = keras_tuner.HyperParameters()
# # lacuna_2b_model = build_simple_lacuna()
# tuner = keras_tuner.RandomSearch(hypermodel=build_simple_lacuna,
#                                  objective='val_accuracy',
#                                  max_trials=2,
#                                  executions_per_trial=3,
#                                  overwrite=True,
#                                  directory=config.TUNERDIR,
#                                  project_name='lacuna')
#
# tuner.search_space_summary()
#
# tuner.search(lac_int_train_ds, epochs=5, validation_data=lac_int_dev_ds)

# get best model
# models = tuner.get_best_models(num_models=2)
# best_model = models[0]
# best_model.build(input_shape=())
# best_model.summary()

# get best hyperparameters
# best_hps = tuner.get_best_hyperparameters(5)
# print(best_hps[0])

# build a model to train
# model = build_simple_lacuna(best_hps[0])
model = build_simple_lacuna()
# model = build_dense_lacuna(config.VOCABSIZE, config.EMBEDDIM, config.DENSEDIM, config.OUTPUTDIM)
model.summary()

print()
# tuner.results_summary()
# plot model
plot_model(model, config.lacuna_2b_architecture_path, show_shapes=True)

# training
callbacks = [
    keras.callbacks.ModelCheckpoint(config.serialized_model_path, monitor='val_loss', mode='min', save_best_only=True,
                                    verbose=1),
    TrainMonitor(os.path.join(config.lacuna_2b_training_progress_path, 'lacuna_2b_training_progress.png')),
    LearningRateScheduler(opt.step_decay),
    keras.callbacks.TensorBoard(log_dir=config.LOGDIR, histogram_freq=1)
]

lacuna_2b_model_history = model.fit(train_dataset,
                                    validation_data=dev_dataset,
                                    epochs=config.EPOCHS,
                                    batch_size=config.BATCHSIZE,
                                    callbacks=callbacks)
