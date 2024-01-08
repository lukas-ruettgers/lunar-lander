import gym
from gym import wrappers
import numpy as np
from collections import deque
from deepqn import DeepQN
from ppo import PPO
from actorcritic import ActorCritic

# -- EXPERIMENT --
FIGURE_FOLDER = "./report/figures/"
TRAINING = False

# -- HYPERPARAMETERS --
LEARNING_RATE = [0.01, 0.001, 0.0001]
DISCOUNT_FACTOR = [0.9, 0.99, 0.999]

"""
Here are the values of this constant in order to achieve a proper balance of exploitation versus exploration 
at 5,000 episodes:

* 0.99910 - 99.99% exploitation + 0.01% exploration
* 0.99941 - 99.95% exploitation + 0.05% exploration
* 0.99954 - 99.90% exploitation + 0.10% exploration
* 0.99973 - 99.75% exploitation + 0.25% exploration
* 0.99987 - 99.50% exploitation + 0.50% exploration
"""
EPSILON_DECAY = [0.99910, 0.99941, 0.99954, 0.99973, 0.99987]

LEARNING_EPISODES = 5000
TESTING_EPISODES = 100
REPLAY_BUFFER_SIZE = 250000
REPLAY_BUFFER_BATCH_SIZE = 32

# -- LUNAR ENV PARAMS 
MINIMUM_REWARD = -250
STATE_SIZE = 8
NUMBER_OF_ACTIONS = 4

if __name__ == "__main__":
	np.set_printoptions(precision=4)

	env = gym.make("LunarLander-v2")
	average_reward = deque(maxlen=100)

	agent = DeepQN(TRAINING, LEARNING_RATE[2], DISCOUNT_FACTOR[1], EPSILON_DECAY[1])

	print("Alpha: %.4f Gamma: %.3f Epsilon %.5f" % (agent.learning_rate, agent.discount_factor, agent.epsilon_decay))
	
	for episode in range(LEARNING_EPISODES if TRAINING else TESTING_EPISODES):
		current_reward = 0

		s = env.reset()

		for t in range(1000):
			if not TRAINING: 
				env.render()

			a = agent.act(s)
			s_, r, is_terminal, info = env.step(a)

			current_reward += r

			agent.store(s, a, r, s_, is_terminal)
			agent.optimize(s, a, r, s_, is_terminal)

			s = s_

			if is_terminal or current_reward < MINIMUM_REWARD:
				break

		agent.update()

		average_reward.append(current_reward)

		print("%i, %.2f, %.2f, %.2f" % (episode, current_reward, np.average(average_reward), agent.epsilon))

	env.close()
	agent.close()