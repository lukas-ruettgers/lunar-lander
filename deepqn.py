import numpy as np
import random
from collections import deque
import tensorflow as tf

class DeepQN:
	def __init__(self, 
			  training, 
			  learning_rate, 
			  discount_factor, 
			  epsilon_decay,
			  replay_buffer_size,
			  state_size,
			  num_actions,
			  batch_size,
			  ):
		self.training = training
		self.learning_rate = learning_rate
		self.discount_factor = discount_factor
		self.epsilon_decay = epsilon_decay
		self.epsilon = 1.0 if self.training else 0.0
		self.replay_buffer = deque(maxlen=replay_buffer_size)
		self.state_size = state_size
		self.num_actions = num_actions
		self.batch_size = batch_size

		self._create_networks()

		self.saver = tf.train.Saver()

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		if not training:
			self._load_weights()

	def act(self, s):
		if not self.training or np.random.rand() > self.epsilon:
			return np.argmax(self._Q(np.reshape(s, [1, self.state_size]))[0])

		return np.random.choice(self.num_actions)

	def store(self, s, a, r, s_, is_terminal):
		if self.training:
			self.replay_buffer.append((np.reshape(s, [1, self.state_size]), a, r, np.reshape(s_, [1, self.state_size]), is_terminal))

	def optimize(self, s, a, r, s_, is_terminal):
		if self.training and len(self.replay_buffer) > self.batch_size:
			batch = np.array(random.sample(list(self.replay_buffer), self.batch_size))
			s = np.vstack(batch[:, 0])
			a = np.array(batch[:, 1], dtype=int)
			r = np.array(batch[:, 2], dtype=float)
			s_ = np.vstack(batch[:, 3])

			non_terminal_states = np.where(batch[:, 4] == False)

			if len(non_terminal_states[0]) > 0:
				a_ = np.argmax(self._Q(s_)[non_terminal_states, :][0], axis=1)
				r[non_terminal_states] += np.multiply(self.discount_factor, self._Q_target(s_)[non_terminal_states, a_][0])

			y = self._Q(s)
			y[range(self.batch_size), a] = r
			self._optimize(s, y)

	def update(self): 
		if self.training:
			Q_W1, Q_W2, Q_W3, Q_b1, Q_b2, Q_b3 = self._get_variables("Q")
			Q_target_W1, Q_target_W2, Q_target_W3, Q_target_b1, Q_target_b2, Q_target_b3 = self._get_variables("Q_target")
			self.sess.run([Q_target_W1.assign(Q_W1), Q_target_W2.assign(Q_W2), Q_target_W3.assign(Q_W3), Q_target_b1.assign(Q_b1), Q_target_b2.assign(Q_b2), Q_target_b3.assign(Q_b3)])

		if self.epsilon > 0.01:
			self.epsilon *= self.epsilon_decay

	def _optimize(self, s, y):
		optimizer, loss, Q_network = self.sess.run([self.optimizer, self.loss, self.Q_network], {self.Q_X: s, self.Q_y: y})

	def _Q(self, s):
		return self.sess.run(self.Q_network, {self.Q_X: s})

	def _Q_target(self, s):
		return self.sess.run(self.Q_target_network, {self.Q_target_X: s})

	def _create_networks(self):
		with tf.variable_scope("Q", reuse=tf.AUTO_REUSE):
			self.Q_X, self.Q_network = self._create_network()
			self.Q_y = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32, name="y")

		with tf.name_scope("loss"):
			self.loss = tf.reduce_mean(tf.squared_difference(self.Q_y, self.Q_network))

		with tf.name_scope("train"):
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

		with tf.variable_scope("Q_target"):
			self.Q_target_X, self.Q_target_network = self._create_network()

	def _create_network(self):
		X = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name="X")

		layer1 = tf.contrib.layers.fully_connected(X, 32, activation_fn=tf.nn.relu)
		layer2 = tf.contrib.layers.fully_connected(layer1, 32, activation_fn=tf.nn.relu)
		network = tf.contrib.layers.fully_connected(layer2, self.num_actions, activation_fn=None)

		return X, network
