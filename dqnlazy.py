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
            a = tf.matmul(self.W, X) + self.b
        else:
            a = tf.matmul(self.W, X)
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


        self.X = tf.placeholder(tf.float32, (None, n_inputs), name = 'X')
        self.G = tf.placeholder(tf.float32, (None, ), name = 'G')
        self.actions = tf.placeholder(tf.int32, (None,), name = 'actions')


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
        for p,q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)


    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict = {self.X : X})

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
        targets = [r + self.gamma*next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]





























