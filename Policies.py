import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from matplotlib import pyplot as plt
tf.keras.backend.set_floatx('float64')
import os

class DqnAgent(object):

    def init_Q_network(self, model_param):
        """
        Builds the Q-network as a keras model.
        """
        num_inputs = model_param["num_inputs"]
        num_outputs = model_param["num_outputs"]
        num_hidden_1 = model_param["num_neurons"][0]
        num_hidden_2 = model_param["num_neurons"][1]
        af = model_param["af"]

        input_layer = layers.Input(shape=(num_inputs,))

        dense_layer_1 = layers.Dense(num_hidden_1, activation=af, kernel_initializer="HeUniform")(input_layer)
        dense_layer_2 = layers.Dense(num_hidden_2, activation=af, kernel_initializer="HeUniform")(dense_layer_1)

        value_layer, advantage_layer = layers.Lambda(lambda w: tf.split(w, 2, 1))(dense_layer_2)

        value_layer = layers.Dense(1)(value_layer)
        advantage_layer = layers.Dense(num_outputs)(advantage_layer)

        # Combine streams into Q-Values
        reduce_mean_layer = layers.Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))

        q_vals_output = layers.Add()([value_layer, layers.Subtract()([advantage_layer, reduce_mean_layer(advantage_layer)])])

        model = Model(input_layer, q_vals_output)
        model.compile(model_param["optimiser"], loss=model_param["loss_func"])

        keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

        return model

    def __init__(self, training_param, model_param, buffer):
        model = self.init_Q_network(model_param)
        self.training_param = training_param
        self.replay_buffer = buffer
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
        self.DQN = model
        self.target_dqn = model

    def calc_epsilon(self, episode, evaluation=False):
        """Get the appropriate epsilon value given the episode
        Arguments:
            episode: Episode during training
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            The appropriate epsilon value
        """
        if evaluation:
            return self.eps_evaluation
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
        if np.random.rand(1) < eps:
            return np.random.randint(0, 2)

        # Otherwise, query the DQN for an action
        q_vals = self.DQN.predict(state)
        return q_vals.argmax()

    def update_target_network(self):
        """Update the target Q network"""
        self.target_dqn.set_weights(self.DQN.get_weights())

    def add_experience(self, state, action, reward, terminal):
        """Wrapper function for adding an experience to the Agent's replay buffer"""
        self.replay_buffer.add_experience(state, action, reward, terminal)

    def learn(self, episode, priority_scale=0.7):
        """Sample a batch and use it to improve the DQN
        Arguments:
            episode: Used for calculating importances
            priority_scale: How much to weight priorities when sampling the replay buffer. 0 = completely random, 1 = completely based on priority
        Returns:
            The loss between the predicted and target Q as a float
        """

        if self.use_per:
            (states, actions, rewards, new_states,
             terminal_flags), importance, indices = self.replay_buffer.get_minibatch(batch_size=self.batch_size,
                                                                                     priority_scale=priority_scale)
            importance = importance ** (1 - self.calc_epsilon(episode))
        else:
            states, actions, rewards, new_states, terminal_flags = self.replay_buffer.get_minibatch(
                batch_size=self.batch_size, priority_scale=priority_scale)

        states = np.squeeze(states)
        new_states = np.squeeze(new_states)

        # Main DQN estimates best action in new states
        arg_q_max = self.DQN.predict(new_states).argmax(axis=1)

        # Target DQN estimates q-vals for new states
        future_q_vals = self.target_dqn.predict(new_states)
        double_q = future_q_vals[range(self.batch_size), arg_q_max]

        # TODO remove terminal flags and set next_state=0 if experience over (terminal) -> Q =0
        # Calculate targets (bellman equation)
        target_q = rewards + (self.gamma * double_q * (1 - terminal_flags))

        # Use targets to calculate loss (and use loss to calculate gradients)
        with tf.GradientTape() as tape:
            q_values = self.DQN(states)

            one_hot_actions = tf.keras.utils.to_categorical(actions, 2,
                                                            dtype=np.float32)  # using tf.one_hot causes strange errors
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            error = Q - target_q
            loss = self.training_param["loss_func"](target_q, Q)

            if self.use_per:
                # Multiply the loss by importance, so that the gradient is also scaled.
                # The importance scale reduces bias against situataions that are sampled
                # more frequently.
                loss = tf.reduce_mean(loss * importance)

        model_gradients = tape.gradient(loss, self.DQN.trainable_variables)
        self.DQN.optimizer.apply_gradients(zip(model_gradients, self.DQN.trainable_variables))

        if self.use_per:
            self.replay_buffer.set_priorities(indices, error)

        return float(loss.numpy()), error

    def save_weights(self, file_loc):
        # Save DQN and target DQN
        self.DQN.save(file_loc)
        self.target_dqn.save(file_loc)

        # Save replay buffer
        self.replay_buffer.save(file_loc)

    def load_weights(self, file_loc):
        # Load DQNs
        self.DQN = tf.keras.models.load_model(file_loc)
        self.target_dqn = tf.keras.models.load_model(file_loc)

