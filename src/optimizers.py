import numpy as np
import config


def step_decay(epoch):
    # initialize the base initial learning rate, drop factor, and
    # epochs to drop every
    initAlpha = config.LEARNINGRATE
    factor = config.DROPFACTOR
    dropEvery = config.DROPEVERY

    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    # return the learning rate
    return float(alpha)
