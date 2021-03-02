import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.config.experimental.set_visible_devices([], "GPU")

from Policies import *
from Logging import *


class Main(object):
    def __init__(self, env, agent, buffer, evaluation):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.evaluation = evaluation    # Flag used to change epsilon for evaluation instead of training.

    def train_DQN(self):
        """ Main training loop """
        rewards_history = []
        running_reward_history = []
        loss_list = []
        mean_loss_list = []
        epsilon_list = []
        mean_epsilon_list = []
        running_reward = 0
        episode = 0
        max_timesteps = self.agent.training_param["max_timesteps"]
        batch_size = self.agent.training_param["batch_size"]
        train_count = 1

        # Loop through all episodes:
        while True:
            episode_reward = 0
            state = self.env.reset()
            state = tf.convert_to_tensor(state)
            state = np.expand_dims(state, 0)

            # Loop through all timesteps in each episode:
            for timestep in range(max_timesteps):

                # if episode == 50:
                #     self.env.render()

                action = self.agent.get_action(train_count, state, evaluation=self.evaluation)
                epsilon_list.append(self.agent.epsilon)

                next_state, reward, done, _ = self.env.step(action)
                next_state = tf.convert_to_tensor(next_state)
                next_state = np.expand_dims(next_state, 0)

                episode_reward += reward

                self.buffer.add_experience((state, action, reward, next_state, done))
                state = next_state

                if timestep % 1 == 0 and len(self.buffer.buffer) >= batch_size + 1000:
                    loss_list.append(self.agent.train_step())
                    train_count += 1  # Used for decreasing epsilon

                if done:
                    break

            # Lists for plotting:
            rewards_history.append(episode_reward)
            mean_loss_list.append(np.mean(loss_list))
            loss_list = []
            mean_epsilon_list.append(np.mean(epsilon_list))
            epsilon_list = []

            # Update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            running_reward_history.append(running_reward)

            episode += 1

            # Log details
            if episode % 10 == 0:
                template = "Running reward: {:.2f} at Episode {}. Loss: {:.2f}"
                print(template.format(running_reward, episode, mean_loss_list[-1]))

            if running_reward >= 1000 or episode == self.agent.training_param["max_num_episodes"]:
                print("Solved at episode {}!".format(episode))

                # Sum rewards and losses during training:
                plt.figure(1)
                plt.plot(np.arange(episode), rewards_history, '.-r')
                plt.plot(np.arange(episode), mean_loss_list, '.-b')  ## NB beginning nan loss is from mean above. Not training.

                # Plot of epsilon evolution during training:
                plt.figure(2)
                plt.plot(np.arange(len(mean_epsilon_list)),mean_epsilon_list, '.-g')
                plt.show()
                break # break the while (episode loop)

    def runSimulation(self, simulated_timesteps):
        """ For running the simulation after training """
        state = env.reset()
        for _ in range(simulated_timesteps):
            env.render()
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            # action to take based on model:
            file_loc = self.agent.DQN_model.model_params["weights_file_loc"]
            self.agent.model.load_weights(file_loc)
            action_probabilities, critic_values = self.agent.DQN_model(state)
            action = np.random.choice(self.agent.DQN_model.model_params["num_outputs"],
                                      p=np.squeeze(action_probabilities))
            # get observation from environment based on action taken:
            observation, reward, done, info = env.step(action)
            state = observation
            if done:
                env.close()
                break


if __name__ == "__main__":
    seed = 42
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    # Parameters used for training
    training_param = {
        "seed": seed,
        "gamma": 0.95,
        "max_timesteps": 1000,
        "max_num_episodes": 100,  # Small for checking plots
        "max_buffer_size": 50000,
        "batch_size": 32,
        "epsilon_max": 1.0,         # Initial epsilon - Exploration
        "epsilon_min": 0.1,        # Final epsilon - Exploitation
        "decay_rate": 1-10**(-4),
        "optimiser": keras.optimizers.Adam(learning_rate=0.0001),
        "loss_func": keras.losses.Huber(),
    }

    # Parameters of the neural network
    model_param = {
        "seed": seed,
        "num_inputs": 4,
        "num_outputs": 2,
        "num_neurons": [50, 50],
        "af": "relu",
        "weights_file_loc": "./model/model_weights"
    }

    env.seed(training_param["seed"])
    tf.random.set_seed(training_param["seed"])
    np.random.seed(training_param["seed"])

    # Create buffer object
    buffer = TrainingBuffer(training_param["max_buffer_size"], training_param["batch_size"])

    DQN_model = DqnNetwork(model_param)
    DQN_agent = DqnAgent(DQN_model, training_param, model_param, buffer)

    # Train
    main = Main(env, DQN_agent, buffer, evaluation=False)
    main.train_DQN()

    # Simulate
    main.evaluation = True
    main.runSimulation(1000)
