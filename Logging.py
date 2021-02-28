from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from collections import deque

class TrainingBuffer(object):
    """
    The training buffer is used to store experiences that are then sampled from uniformly to facilitate
    improved training. The training buffer reduces the correlation between experiences and avoids that
    the network 'forgets' good actions that it learnt previously.
    """

    def __init__(self, max_mem_size):
        self.buffer = deque(maxlen=max_mem_size)

    def add_experience(self, experience):
        """
        Add an experience (s_k, a_k, r_k, s_k+1) to the training buffer.
        """
        self.buffer.append(experience)

    def get_training_samples(self, batch_size):
        index = np.random.choice(np.arange(len(self.buffer)),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]
