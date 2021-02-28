import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used

from CartPole import CartPoleEnv  # CartPoleEnv has been modified to make it more difficult.
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.config.experimental.set_visible_devices([], "GPU")

from Policies import *
from Logging import *


class Main(object):
    def __init__(self, env, agent, buffer):
        self.env = env
        self.agent = agent
        self.buffer = buffer

    def train_DQN(self):
        rewards_history = []
        running_reward_history = []
        loss_list = []
        running_reward = 0
        episode = 0
        max_timesteps = self.agent.training_param["max_timesteps"]
        batch_size = self.agent.training_param["batch_size"]
        # Loop through all episodes:
        while True:
            episode_reward = 0
            state = self.env.reset()
            state = tf.convert_to_tensor(state)
            state = np.expand_dims(state, 0)

            # Loop through all timesteps in each episode:
            for timestep in range(max_timesteps):

                if episode == 50:
                    self.env.render()

                action = self.agent.get_action(episode, state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = tf.convert_to_tensor(next_state)
                next_state = np.expand_dims(next_state, 0)

                episode_reward += reward

                if not done:
                    self.buffer.add_experience((state, action, reward, next_state))
                    state = next_state
                    if timestep % 1 == 0 and len(self.buffer.buffer) > batch_size:
                        self.agent.train_step()
                else:
                    next_state = np.zeros(np.shape(state))
                    self.buffer.add_experience((state, action, reward, next_state))
                    if timestep % 1 == 0 and len(self.buffer.buffer) > batch_size:
                        self.agent.train_step()
                    break

            rewards_history.append(episode_reward)

            # Update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            running_reward_history.append(running_reward)

            # Log details
            episode += 1

            if episode % 10 == 0:
                template = "running reward: {:.2f} at episode {}"
                print(template.format(running_reward, episode))

            if running_reward >= 1000 or episode >= self.agent.training_param["max_num_episodes"]:
                print("Solved at episode {}!".format(episode))
                file_loc = self.agent.model.model_params["weights_file_loc"]
                self.agent.model.save_weights(file_loc)
                break

    def runSimulation(self, simulated_timesteps):
        state = env.reset()
        for _ in range(simulated_timesteps):
            env.render()
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            # action to take based on model:
            file_loc = self.agent.model.model_params["weights_file_loc"]
            self.agent.model.load_weights(file_loc)
            action_probabilities, critic_values = self.agent.model(state)
            action = np.random.choice(self.agent.model.model_params["num_outputs"],
                                      p=np.squeeze(action_probabilities))
            # get observation from environment based on action taken:
            observation, reward, done, info = env.step(action)
            state = observation
            if done:
                env.close()
                break


if __name__ == "__main__":
    seed = 42

    # Set configuration parameters for the whole setup
    training_param = {
        "seed": seed,
        "gamma": 0.99,
        "max_timesteps": 1000,
        "max_num_episodes": 1000,
        "max_buffer_size": 10000,
        "batch_size": 32,
        "use_per": False,
        "epsilon_max": 1.0,         # Initial epsilon - Exploration
        "epsilon_min": 0.01,        # Final epsilon - Exploitation
        "decay_rate": 10**(-2.6),   # Tuned to reach epsilon max in 1000 episodes
        "optimiser": keras.optimizers.Adam(learning_rate=0.0001),
        "loss_func": keras.losses.Huber(),
    }

    model_param = {
        "seed": seed,
        "num_inputs": 4,
        "num_outputs": 2,
        "num_neurons": [100, 100],
        "af": "relu",
        "weights_file_loc": "./model/model_weights"
    }

    env = CartPoleEnv()  # Create the environment
    env.seed(training_param["seed"])
    tf.random.set_seed(training_param["seed"])
    np.random.seed(training_param["seed"])

    buffer = TrainingBuffer(training_param["max_buffer_size"])

    DQN_model = DqnNetwork(model_param)
    DQN_agent = DqnAgent(DQN_model, training_param, model_param, buffer)

    main = Main(env, DQN_agent, buffer)
    main.train_DQN()

    main.runSimulation(1000)
