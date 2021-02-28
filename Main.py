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

    def fill_buffer(self):
        # Fill training buffer with experiences to start training
        batch_size = self.agent.training_param["batch_size"]
        # while len(self.buffer.buffer) < self.trainer.training_param["max_buffer_size"]:

        state = self.env.reset()
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        for i in range(batch_size+1):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            next_state = tf.convert_to_tensor(next_state)
            next_state = tf.expand_dims(next_state, 0)

            if not done:
                self.buffer.add_experience((state, action, reward, next_state))
                state = next_state
            else:
                next_state = np.zeros(np.shape(state))
                self.buffer.add_experience((state, action, reward, next_state))
                state = self.env.reset()
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)
        return state


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
            # state = tf.convert_to_tensor(state)
            state = np.expand_dims(state, 0)
            # Loop through all timesteps in each episode:
            for timestep in range(max_timesteps):

                if episode == 50:
                    self.env.render()

                action = self.agent.get_action(episode, state)

                next_state, reward, done, _ = self.env.step(action)
                # next_state = tf.convert_to_tensor(next_state)
                next_state = np.expand_dims(next_state, 0)


                episode_reward += reward

                # Add experience to replay memory
                self.agent.add_experience(action=action,
                                          state=next_state,
                                          reward=reward,
                                          terminal=done)

                # Update agent
                if self.agent.replay_buffer.count > batch_size:
                    loss, _ = self.agent.learn(episode)
                    loss_list.append(loss)

                # Update target network
                if timestep == range(max_timesteps)[-1]:
                    self.agent.update_target_network()

                # Break the loop when the game is over
                if done:
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
        # # Run until all episodes completed (reward level reached)
        # state = self.fill_buffer()
        #
        # while True:
        #     episode_reward = 0
        #     for timestep in range(max_timesteps):
        #         if episode_count % 50 == 0 and episode_count != 0:
        #             self.env.render()
        #
        #         # explore_prob = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay*decay_step)
        #         if explore_prob[episode_count] > np.random.rand():
        #             action = self.env.action_space.sample()
        #         else:
        #             action = np.argmax(self.trainer.model(state))
        #
        #
        #         next_state, reward, done, _ = self.env.step(action)
        #         next_state = tf.convert_to_tensor(next_state)
        #         next_state = tf.expand_dims(next_state, 0)
        #
        #         episode_reward += reward
        #
        #         if not done:
        #             self.buffer.add_experience((state, action, reward, next_state))
        #             state = next_state
        #             if timestep % 5 == 0:
        #                 self.trainer.train_step()
        #         else:
        #             next_state = np.zeros(np.shape(state))
        #             self.buffer.add_experience((state, action, reward, next_state))
        #             state = self.env.reset()
        #             state = tf.convert_to_tensor(state)
        #             state = tf.expand_dims(state, 0)
        #             if timestep % 5 == 0:
        #                 self.trainer.train_step()
        #             break
        #
        #
        #     # Update running reward to check condition for solving
        #     running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        #     running_reward_history.append(running_reward)
        #
        #     # Log details
        #     episode_count += 1
        #     decay_step += 1
        #     if episode_count % 10 == 0:
        #         template = "running reward: {:.2f} at episode {}"
        #         print(template.format(running_reward, episode_count))
        #
        #     if running_reward >= 1000 or episode_count >= self.trainer.training_param["max_num_episodes"]:
        #         print("Solved at episode {}!".format(episode_count))
        #         file_loc = self.agent.model.model_params["weights_file_loc"]
        #         self.agent.model.save_weights(file_loc)
        #         break

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
        "optimiser": keras.optimizers.Adam(learning_rate=0.001),
        "loss_func": keras.losses.Huber(),
    }

    model_param = {
        "seed": seed,
        "num_inputs": 4,
        "num_outputs": 2,
        "num_neurons": [100, 100],
        "af": "relu",
        "weights_file_loc": "./model/model_weights",
        "optimiser": keras.optimizers.Adam(learning_rate=0.01),
        "loss_func": keras.losses.Huber()
    }

    env = CartPoleEnv()  # Create the environment
    env.seed(training_param["seed"])
    tf.random.set_seed(training_param["seed"])
    np.random.seed(training_param["seed"])

    buffer = PerTrainingBuffer(size=training_param["max_buffer_size"], use_per=training_param["use_per"])

    DQN_agent = DqnAgent(training_param, model_param, buffer)

    main = Main(env, DQN_agent, buffer)
    main.train_DQN()

    main.runSimulation(1000)
