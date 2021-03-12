from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random
from collections import deque
import tensorflow as tf

class TrainingBuffer(object):
    """
    The training buffer is used to store experiences that are then sampled from uniformly to facilitate
    improved training. The training buffer reduces the correlation between experiences and avoids that
    the network 'forgets' good actions that it learnt previously.
    """

    def __init__(self, max_mem_size, batch_size, use_per):
        self.use_per = use_per
        self.max_mem_size = max_mem_size
        self.buffer = deque(maxlen=max_mem_size)
        self.batch_size = batch_size
        self.priority_scale = 0.7

    def add_experience(self, experience):
        """
        Add an experience (s_k, a_k, r_k, s_k+1) to the training buffer.
        """
        self.buffer.append(experience)

    def get_training_samples(self):
        """Returns a minibatch"""
        mini_batch = random.sample(self.buffer, self.batch_size)
        states = tf.squeeze(tf.convert_to_tensor([each[0] for each in mini_batch], dtype=np.float32))
        actions = tf.squeeze(tf.convert_to_tensor(np.array([each[1] for each in mini_batch], dtype=np.float32)))
        rewards = tf.squeeze(tf.convert_to_tensor(np.array([each[2] for each in mini_batch], dtype=np.float32)))
        next_states = tf.squeeze(tf.convert_to_tensor(np.array([each[3] for each in mini_batch], dtype=np.float32)))
        done = tf.cast([each[4] for each in mini_batch], dtype=tf.float32)
        return states, actions, rewards, next_states, done

    def is_buffer_min_size(self):
        return len(self.buffer) >= self.batch_size
