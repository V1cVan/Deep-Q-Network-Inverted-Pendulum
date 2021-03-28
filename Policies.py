import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import random
from matplotlib import pyplot as plt
tf.keras.backend.set_floatx('float64')
import os
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

class DqnNetwork(keras.Model):
    """
    Builds the Q-network as a keras model.
    """
    def __init__(self, model_param):
        super(DqnNetwork, self).__init__()
        self.model_params = model_param
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
                                 name="DDQN_basic")

        keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True)

    @tf.function
    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        y = tf.cast(self.model(inputs), dtype=tf.dtypes.float32)
        return y


class DuellingDqnNetwork(keras.Model):
    """
    Builds the Q-network as a keras model.
    """

    def __init__(self, model_param):
        super(DuellingDqnNetwork, self).__init__()
        self.model_params = model_param
        num_inputs = model_param["num_inputs"]
        num_outputs = model_param["num_outputs"]
        num_hidden_1 = model_param["num_neurons"][0]
        num_hidden_2 = model_param["num_neurons"][1]
        af = model_param["af"]

        input_layer = layers.Input(shape=(num_inputs,))

        dense_layer_1 = layers.Dense(num_hidden_1, activation=af)(input_layer)
        dense_layer_2 = layers.Dense(num_hidden_2, activation=af)(dense_layer_1)

        value_layer, advantage_layer = layers.Lambda(lambda w: tf.split(w, 2, 1))(dense_layer_2)

        value_layer = layers.Dense(1)(value_layer)
        advantage_layer = layers.Dense(num_outputs)(advantage_layer)

        reduce_mean_layer = layers.Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))

        output_layer = layers.Add()([value_layer, layers.Subtract()([advantage_layer, reduce_mean_layer(advantage_layer)])])

        self.model = keras.Model(inputs=input_layer,
                                 outputs=output_layer,
                                 name="DDQN_basic")

        keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True)

    @tf.function
    def call(self, inputs: tf.Tensor):
        """ Returns the output of the model given an input. """
        y = tf.cast(self.model(inputs), dtype=tf.dtypes.float32)
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

        # DDQN networks
        self.DQN_model = model
        self.DQN_target = model

    def update_target_net(self):
        self.DQN_target.set_weights(self.DQN_model.get_weights())

    def calc_epsilon(self, epsilon_decay_count, evaluation=False):
        """ Calculate epsilon based on the training counter in the training loop. """
        if evaluation:  # (if not training)
            self.epsilon = 0
            return self.epsilon
        elif not self.buffer.is_buffer_min_size() :  # (if buffer isnt full)
            self.epsilon = 1
            return self.epsilon
        else:
            if epsilon_decay_count > 1 and self.epsilon > self.eps_final:
                self.epsilon = self.epsilon * self.decay
            return self.epsilon


    def get_action(self, epsilon_decay_count, state, evaluation=False):
        """ Get action to be passed to the simulator. """
        eps = self.calc_epsilon(epsilon_decay_count, evaluation)
        # With chance epsilon, take a random action
        if np.random.rand() < eps and evaluation is False:
            return np.random.randint(0, 2)
        else:
            # Otherwise, query the DQN for an action
            q_vals = self.DQN_model(state)
            return np.argmax(q_vals, axis=1)[0]

    def add_experience(self, experience):
        if self.training_param["use_per"]:
            # Calculate the TD-error for the Prioritised Replay Buffer
            states, actions, rewards, next_states, done = experience
            states = tf.convert_to_tensor(states, dtype=np.float32)
            actions = tf.squeeze(tf.convert_to_tensor(actions, dtype=np.float32))
            rewards = tf.squeeze(tf.convert_to_tensor(rewards, dtype=np.float32))
            next_states = tf.convert_to_tensor(next_states, dtype=np.float32)
            done = tf.cast(done, dtype=tf.float32)
            td_error = self.compute_td_error(states=states,
                                             rewards=rewards,
                                             next_states=next_states,
                                             done=done)
            self.buffer.add_experience(td_error, (states, actions, rewards, next_states, done))
        else:
            self.buffer.add_experience(experience)


    @tf.function
    # TODO calculate TD error in the loss
    def compute_td_error(self,
                         states: tf.Tensor,
                         rewards: tf.Tensor,
                         next_states: tf.Tensor,
                         done: tf.Tensor ):
        ones = tf.ones(tf.shape(done), dtype=tf.dtypes.float32)
        target_Q = self.DQN_target(next_states)
        target_output = rewards + (ones - done) * (self.gamma * tf.reduce_max(target_Q, axis=1))
        predicted_Q = self.DQN_model(states)
        predicted_output = tf.reduce_max(predicted_Q, axis=1)
        return target_output - predicted_output


    def train_step(self):
        """ Training step. """
        # Sample mini-batch from memory
        if self.training_param["use_per"]:
            states, actions, rewards, next_states, done, idxs, is_weight = self.buffer.get_training_samples()
            one_hot_actions = tf.keras.utils.to_categorical(actions, 2, dtype=np.float32)
            batch_reward, loss, td_error = self.run_tape(
                states=states,
                actions=one_hot_actions,
                rewards=rewards,
                next_states=next_states,
                done=done,
                is_weight=is_weight
            )
        else:
            states, actions, rewards, next_states, done = self.buffer.get_training_samples()
            one_hot_actions = tf.keras.utils.to_categorical(actions, 2, dtype=np.float32)
            batch_reward, loss, td_error = self.run_tape(
                states=states,
                actions=one_hot_actions,
                rewards=rewards,
                next_states=next_states,
                done=done
            )
        self.buffer.update(idxs, td_error)
        return batch_reward, loss

    #@tf.function
    def run_tape(self,
                 states: tf.Tensor,
                 actions: tf.Tensor,
                 rewards: tf.Tensor,
                 next_states: tf.Tensor,
                 done: tf.Tensor,
                 is_weight: tf.Tensor = None):

        ones = tf.ones(tf.shape(done), dtype=tf.dtypes.float32)

        target_Q = self.DQN_target(next_states)
        target_output = rewards + (ones - done) * (self.gamma * tf.reduce_max(target_Q, axis=1))

        # Testing the standardisation of expected returns - Showed significant performance in another repo.
        if self.training_param["standardise_returns"]:
            eps = np.finfo(np.float32).eps.item()
            target_output = target_output - tf.math.reduce_mean(target_output) / (
                        tf.math.reduce_std(target_output) + eps)

        with tf.GradientTape() as tape:
            predicted_Q = self.DQN_model(states)

            predicted_output = tf.reduce_sum(tf.multiply(predicted_Q, actions), axis=1)


            td_error = target_output - predicted_output
            loss_value = tf.reduce_mean(tf.square(td_error))
            if is_weight is not None:
                loss_value = tf.reduce_mean(loss_value * is_weight)

            # loss_value = self.training_param["loss_func"](target_output, predicted_output)

        grads = tape.gradient(loss_value, self.DQN_model.trainable_variables)
        # Clip gradients
        if self.training_param["clip_gradients"]:
            norm = self.training_param["clip_norm"]
            grads = [tf.clip_by_norm(g, norm)
                     for g in grads]

        self.training_param["optimiser"].apply_gradients(zip(grads, self.DQN_model.trainable_variables))
        sum_reward = tf.math.reduce_sum(rewards)
        batch_reward = sum_reward/self.buffer.get_size()

        return batch_reward, loss_value, td_error