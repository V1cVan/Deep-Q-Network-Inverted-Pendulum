import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used

import gym
import tensorflow as tf
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
        model_update_counter = 0
        trained = False
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

                action = self.agent.get_action(train_count, state, evaluation=self.evaluation)
                epsilon_list.append(self.agent.epsilon)

                next_state, reward, done, _ = self.env.step(action)
                next_state = tf.convert_to_tensor(next_state)
                next_state = np.expand_dims(next_state, 0)

                episode_reward += reward

                self.agent.add_experience((state, action, reward, next_state, done))
                state = next_state

                # TODO add update step to the PER
                if self.buffer.is_buffer_min_size():
                    model_update_counter += 1
                    if model_update_counter % training_param["target_update_rate"] == 0:
                        self.agent.update_target_net()
                        # print("Updated target net.")
                    if model_update_counter % training_param["model_update_rate"] == 0:
                        batch_reward, loss = self.agent.train_step()
                        trained = True
                        loss_list.append(loss)
                        train_count += 1  # Used for decreasing epsilon


                if done:
                    break

            # Lists for plotting:
            rewards_history.append(episode_reward)
            mean_epsilon_list.append(np.mean(epsilon_list))
            epsilon_list = []

            # Update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            running_reward_history.append(running_reward)

            episode += 1

            # Log details
            if episode % 5 == 0:
                template = "Running reward: {:.2f} at Episode {}. Loss: {:.2f}. Epsilon: {:.2f}."
                if trained == True:
                    print(template.format(running_reward, episode, loss_list[-1], mean_epsilon_list[-1]))
                    trained == False

            if running_reward == 350 or episode == self.agent.training_param["max_num_episodes"]:
                print("Solved at episode {}!".format(episode))
                self.agent.DQN_model.save_weights(self.agent.DQN_model.model_params["weights_file_loc"])
                break

    def runSimulation(self, simulated_timesteps):
        """ For running the simulation after training """
        state = env.reset()
        for _ in range(simulated_timesteps):
            env.render()
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            # action to take based on model:
            file_loc = self.agent.DQN_model.model_params["weights_file_loc"]
            self.agent.DQN_model.load_weights(file_loc)
            action = self.agent.get_action(epsilon_decay_count=0, state=state, evaluation=True)
            # get observation from environment based on action taken:
            observation, reward, done, info = env.step(action)
            state = observation
            if done:
                env.close()
                break


if __name__ == "__main__":
    SEED = 42
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    # Training parameters
    GAMMA = 0.99
    MAX_TIMESTEPS = 1000
    MAX_EPISODES = 30000
    BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    EPSILON_MAX = 1.0
    EPSILON_MIN = 0.1
    DECAY_RATE = 0.99994#0.999997   # Epsilon decay rate per training step taken
    MODEL_UPDATE_RATE = 5  # How many simulation steps are taken before the network weights are updated
    TARGET_UPDATE_RATE = 100000*MODEL_UPDATE_RATE
    CLIP_GRADIENTS = True
    CLIP_NORM = 2
    STANDARDISE_RETURNS = False
    LEARN_RATE = 0.0008
    OPTIMISER = keras.optimizers.Adam(learning_rate=LEARN_RATE)
    LOSS_FUNC = tf.losses.Huber()
    USE_PER = True          # Toggle to sample according to priority instead of uniformly
    USE_DUELLING = True     # Enable duelling network architecture

    # Model parameters
    NUM_INPUTS = env.observation_space.shape[0]
    NUM_OUTPUTS = env.action_space.n
    NUM_NEURONS = [200, 200]
    AF = "relu"
    WEIGHTS_FILE_LOC = "./model/model_weights"

    training_param = {
        "seed": SEED,
        "gamma": GAMMA,
        "max_timesteps": MAX_TIMESTEPS,
        "max_num_episodes": MAX_EPISODES,  # Small for checking plots
        "max_buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "epsilon_max": EPSILON_MAX,         # Initial epsilon - Exploration
        "epsilon_min": EPSILON_MIN,        # Final epsilon - Exploitation
        "decay_rate": DECAY_RATE,
        "optimiser": OPTIMISER,
        "loss_func": LOSS_FUNC,
        "model_update_rate": MODEL_UPDATE_RATE,
        "target_update_rate": TARGET_UPDATE_RATE,
        "standardise_returns": STANDARDISE_RETURNS,
        "clip_gradients": CLIP_GRADIENTS,
        "clip_norm": CLIP_NORM,
        "use_per": USE_PER,
        "use_dueling": USE_DUELLING
    }
    model_param = {
        "seed": SEED,
        "num_inputs": NUM_INPUTS,
        "num_outputs": NUM_OUTPUTS,
        "num_neurons": NUM_NEURONS,
        "af": AF,
        "weights_file_loc": WEIGHTS_FILE_LOC
    }

    env.seed(training_param["seed"])
    tf.random.set_seed(training_param["seed"])
    np.random.seed(training_param["seed"])
    random.seed(training_param["seed"])

    # Create buffer object
    if USE_PER:
        buffer = PerTrainingBuffer(buffer_size=BUFFER_SIZE,
                                   batch_size=BATCH_SIZE)
    else:
        buffer = TrainingBuffer(max_mem_size=BUFFER_SIZE,
                                batch_size=BATCH_SIZE)

    if USE_DUELLING:
        DQN_model = DuellingDqnNetwork(model_param)
    else:
        DQN_model = DqnNetwork(model_param)

    DQN_agent = DqnAgent(DQN_model, training_param, model_param, buffer)

    # Train
    main = Main(env=env, agent=DQN_agent, buffer=buffer, evaluation=False)
    main.train_DQN()

    # Simulate
    main.evaluation = True
    main.runSimulation(1000)
