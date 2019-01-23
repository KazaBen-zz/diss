import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers


class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
        self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
        self.params = [self.W]
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)
        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)


class DQN:
    def __init__(self, n_inputs, n_outputs, hidden_layer_sizes, gamma, max_experiences, min_experiences, batch_size):
        self.n_outputs = n_outputs

        # Creating graph
        self.layers = []

        # TODO: WTF is this
        M1 = n_inputs
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # Output layer
        layer = HiddenLayer(M1, n_outputs, lambda x: x)
        self.layers.append(layer)

        # Collect params
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        self.X = tf.placeholder(tf.float32, (None, n_inputs), name='X')
        self.G = tf.placeholder(tf.float32, (None,), name='G')
        self.actions = tf.placeholder(tf.int32, (None,), name='actions')

        # Calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z
        self.predict_op = Y_hat

        selected_action_values = tf.reduce_sum(
            Y_hat * tf.one_hot(self.actions, n_outputs),
            reduction_indices=[1]
        )

        cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)

        # Create replay memory
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.gamma = gamma

    def set_session(self, session):
        self.session = session

    def copy_from(self, other):
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def train(self, target_network):
        if len(self.experience['s']) < self.min_experiences:
            return

        idx = np.random.choice(len(self.experience['s']), self.batch_size, False)

        states = [self.experience['s'][i] for i in idx]
        actions = [self.experience['a'][i] for i in idx]
        rewards = [self.experience['r'][i] for i in idx]
        next_states = [self.experience['s2'][i] for i in idx]
        dones = [self.experience['done'][i] for i in idx]

        next_Q = np.max(target_network.predict(next_states), axis=1)
        targets = [r + self.gamma * next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

        self.session.run(
            self.train_op,
            feed_dict={
                self.X: states,
                self.G: targets,
                self.actions: actions
            }
        )

    def add_experience(self, s, a, r, s2, done):
        if len(self.experience['s']) >= self.max_experiences:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)
            self.experience['done'].pop(0)
        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['s2'].append(s2)
        self.experience['done'].append(done)


    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.n_outputs)
        else:
            X = np.atleast_2d(x)
            return np.argmax(self.predict(X)[0])


def play_one(env, model, tmodel, eps, gamma, copy_period):
    state = env.reset()
    done = False
    total_reward = 0
    iters = 0
    while not done and iters < 2000:
        action = model.sample_action(state, eps)
        prev_state = state
        state, reward, done, _ = env.step(action)

        total_reward += reward
        if done:
            reward = -200

        model.add_experience(prev_state, action, reward, state, done)
        model.train(tmodel)

        if iters % copy_period == 0:
            tmodel.copy_from(model)

    return total_reward


env = gym.make('CartPole-v0')
gamma = 0.99
copy_period = 50

n_inputs = len(env.observation_space.sample())
n_ouputs = env.action_space.n
sizes = [200, 200]
model = DQN(n_inputs, n_ouputs, sizes, gamma, 10000, 100, 32)
tmodel = DQN(n_inputs, n_ouputs, sizes, gamma, 10000, 100, 32)
init = tf.global_variables_initializer()
session = tf.InteractiveSession()
session.run(init)
model.set_session(session)
tmodel.set_session(session)

N = 500
total_rewards = np.empty(N)
costs = np.empty(N)
for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    total_reward = play_one(env, model, tmodel, eps, gamma, copy_period)
    total_rewards[n] = total_reward

    if n%100 == 0:
        print("episode", n, "total reward", total_reward, "eps", eps, "avg last 100", total_rewards[max(0, n-100):(n+1)].mean())

    print("avg reward for last 100 episodes:", total_rewards[-100:].mean())
    print("total steps:", total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    # plot_running_avg(total_rewards)

