import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from matplotlib import pyplot as plt
tf.keras.backend.set_floatx('float64')
import os

class DqnNetwork(keras.Model):
    """
    Builds the Q-network as a keras model.
    """
    def __init__(self, model_param):
        super(DqnNetwork, self).__init__()

        num_inputs = model_param["num_inputs"]
        num_outputs = model_param["num_outputs"]
        num_hidden_1 = model_param["num_neurons"][0]
        num_hidden_2 = model_param["num_neurons"][1]
        af = model_param["af"]

        input_layer = layers.Input(shape=(num_inputs,))

        dense_layer_1 = layers.Dense(num_hidden_1, activation=af)(input_layer)
        dense_layer_2 = layers.Dense(num_hidden_2, activation=af)(dense_layer_1)
        output_layer = layers.Dense(num_outputs, activation="linear")(dense_layer_2)

        self.model = keras.Model(inputs=input_layer,
                                 outputs=output_layer,
                                 name="DQN_basic")

        keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True)

    @tf.function
    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        y = self.model(inputs)
        return y


class DqnAgent(keras.models.Model):

    def __init__(self, model, training_param, model_param, buffer):
        super(DqnAgent, self).__init__()
        self.training_param = training_param
        self.buffer = buffer
        self.gamma = training_param["gamma"]

        # Replay buffer
        self.batch_size = training_param["batch_size"]

        # Epsilon information
        self.eps_initial = training_param["epsilon_max"]
        self.eps_final = training_param["epsilon_min"]
        self.decay = training_param["decay_rate"]
        self.epsilon = self.eps_initial

        # DQN network
        self.DQN_model = model


    def calc_epsilon(self, epsilon_decay_count, evaluation=False):
        """ Calculate epsilon based on the training counter in the training loop. """
        if evaluation:  # (if not training)
            self.epsilon = 0
            return self.epsilon
        elif not self.buffer.is_buffer_min_size():  # (if buffer isnt full)
            self.epsilon = 1
            return self.epsilon
        else:
            if epsilon_decay_count > 1 and self.epsilon > self.eps_final:
                self.epsilon = self.epsilon*self.decay
            return self.epsilon


    def get_action(self, epsilon_decay_count, state, evaluation=False):
        """ Get action to be passed to the simulator. """
        eps = self.calc_epsilon(epsilon_decay_count, evaluation)
        # With chance epsilon, take a random action
        if np.random.rand() < eps:
            return np.random.randint(0, 2)
        else:
            # Otherwise, query the DQN for an action
            q_vals = self.DQN_model(state)
            return np.argmax(q_vals, axis=1)[0]


    def train_step(self):
        """ Training step. """
        gamma = self.training_param["gamma"]

        # Sample mini-batch from memory
        mini_batch = self.buffer.get_training_samples()
        states = tf.squeeze(tf.convert_to_tensor([each[0] for each in mini_batch]))
        actions = tf.squeeze(tf.convert_to_tensor(np.array([each[1] for each in mini_batch])))
        rewards = tf.squeeze(tf.convert_to_tensor(np.array([each[2] for each in mini_batch])))
        next_states = tf.squeeze(tf.convert_to_tensor(np.array([each[3] for each in mini_batch])))
        done = np.array([each[4] for each in mini_batch])

        with tf.GradientTape() as tape:
            target_Q = self.DQN_model(next_states)
            target_output = rewards + (1-done)*(gamma * np.amax(target_Q, axis=1))

            predicted_Q = self.DQN_model(states)
            one_hot_actions = tf.keras.utils.to_categorical(actions, 2, dtype=np.float32)
            predicted_output = tf.reduce_sum(tf.multiply(predicted_Q, one_hot_actions), axis=1)

            loss_value = self.training_param["loss_func"](target_output, predicted_output)

            grads = tape.gradient(loss_value, self.DQN_model.trainable_variables)

            self.training_param["optimiser"].apply_gradients(zip(grads, self.DQN_model.trainable_variables))

        return loss_value