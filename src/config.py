import os
import matplotlib
import datetime

EMBEDDIM = 128
DENSEDIM = 128
OUTPUTDIM = 17
EPOCHS = 100
LEARNINGRATE = 0.001
MINLEARNINGRATE = 1e-4
MAXLEARNINGRATE = 1e-2
DROPFACTOR = 0.0000025
DROPEVERY = 5
BATCHSIZE = 1024
SEED = 42
NPCALLS = 6
VOCABSIZE = 50000
matplotlib.use("Agg")
base_dir = 'C:\\Users\\Tajr\Desktop\\textlab'
serialized_model_path = 'C:\\Users\\Tajr\Desktop\\textlab\\serialized\\lacuna_2b_model.keras'
lacuna_2b_training_progress_path = 'C:\\Users\\Tajr\Desktop\\textlab\\serialized'
lacuna_2b_architecture_path = 'C:\\Users\\Tajr\Desktop\\textlab\\serialized\\lacuna_2b_architecture.png'
LOGDIR = 'C:\\Users\\Tajr\Desktop\\textlab\\serialized\\logs\\fit\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
TUNERDIR = 'C:\\Users\\Tajr\Desktop\\textlab\\serialized\\tuner'
