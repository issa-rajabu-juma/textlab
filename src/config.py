import os
import matplotlib
import datetime

# general
base_dir = 'C:\\Users\\Tajr\Desktop\\textlab'
matplotlib.use("Agg")

# keras tuner
TUNERDIR = 'C:\\Users\\Tajr\Desktop\\textlab\\serialized\\tuner'
MINLEARNINGRATE = 1e-4
MAXLEARNINGRATE = 1e-2
MINREGULARIZER = 1e-4
MAXREGULARIZER = 1e-2

# tensorboard
LOGDIR = 'C:\\Users\\Tajr\Desktop\\textlab\\serialized\\logs\\fit\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# model
ARCHITECTUREPATH = 'C:\\Users\\Tajr\Desktop\\textlab\\serialized\\architecture\\architecture.png'
KERASMODEL = 'C:\\Users\\Tajr\Desktop\\textlab\\serialized\\checkpoints\\keras\\model.keras'
EMBEDDIM = 32
DENSEDIM = 32
OUTPUTDIM = 17
BATCHSIZE = 512
SEED = 42
NPCALLS = 6
VOCABSIZE = 75000
L2 = 0.000001

# training
PLOTPROGRESS = 'C:\\Users\\Tajr\Desktop\\textlab\\serialized\\progress\\progress.png'
JSONPROGRESS = 'C:\\Users\\Tajr\Desktop\\textlab\\serialized\\progress\\progress.json'
OUTPUTPATH = 'C:\\Users\\Tajr\Desktop\\textlab\\serialized\\checkpoints\\hdf5'
MODEL = 'C:\\Users\\Tajr\Desktop\\textlab\\serialized\\checkpoints\\hdf5\\epoch_15.hdf5'
EPOCHS = 100
LEARNINGRATE = 1e-3
NEWLR = 1e-6
STARTAT = 15
SAVEEVERY = 1
# MODEL = None

# decay
DROPEVERY = 2
DROPFACTOR = 0.5

