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

        dense_layer_1 = layers.Dense(num_hidden_1, activation=af, kernel_initializer="HeUniform")(input_layer)
        dense_layer_2 = layers.Dense(num_hidden_2, activation=af, kernel_initializer="HeUniform")(dense_layer_1)

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
        self.use_per = training_param["use_per"]

        # Epsilon information
        self.eps_initial = training_param["epsilon_max"]
        self.eps_final = training_param["epsilon_min"]
        self.decay = training_param["decay_rate"]
        self.eps_evaluation = self.eps_final

        # DQN
        self.DQN_model = model


    def calc_epsilon(self, episode, evaluation=False):
        """Get the appropriate epsilon value given the episode
        Arguments:
            episode: Episode during training
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            The appropriate epsilon value
        """
        if evaluation:
            return 1
        elif episode == 0:
            return self.eps_initial
        elif episode > 0:
            return 0.1*np.exp(self.decay*episode)

    def get_action(self, episode, state, evaluation=False):
        """Query the DQN for an action given a state
        Arguments:
            episode: episode number
            state: State to give an action for
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            An integer as the predicted move
        """

        # Calculate epsilon based on the frame number
        eps = self.calc_epsilon(episode, evaluation)

        # With chance epsilon, take a random action
        if np.random.random() > eps:
            return np.random.randint(0, 2)
        else:
            # Otherwise, query the DQN for an action
            q_vals = self.DQN_model(state)
            return np.argmax(q_vals, axis=1)[0]

    def train_step(self):
        batch_size = self.training_param["batch_size"]
        gamma = self.training_param["gamma"]

        # Sample mini-batch from memory
        batch = self.buffer.get_training_samples(batch_size)
        states = tf.squeeze(tf.convert_to_tensor([each[0] for each in batch]))
        actions = tf.squeeze(tf.convert_to_tensor(np.array([each[1] for each in batch])))
        rewards = tf.squeeze(tf.convert_to_tensor(np.array([each[2] for each in batch])))
        next_states = tf.squeeze(tf.convert_to_tensor(np.array([each[3] for each in batch])))

        with tf.GradientTape() as tape:
            target_Q = self.DQN_model(next_states)
            target_output = rewards + gamma*np.max(target_Q, axis=1)

            predicted_Q = self.DQN_model(states)
            one_hot_actions = tf.keras.utils.to_categorical(actions, 2,
                                                            dtype=np.float32)  # using tf.one_hot causes strange errors
            predicted_output = tf.reduce_sum(tf.multiply(predicted_Q, one_hot_actions), axis=1)

            loss_value = self.training_param["loss_func"](target_output, predicted_output)

            grads = tape.gradient(loss_value, self.DQN_model.trainable_variables)

            self.training_param["optimiser"].apply_gradients(zip(grads, self.DQN_model.trainable_variables))
