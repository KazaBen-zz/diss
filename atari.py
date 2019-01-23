import copy
import gym
import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from scipy.misc import imresize
from collections import deque

HEIGHT = 81
WIDTH = 72

MIN_EXPERIENCES = 5000
MAX_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000


def preprocess_frame(frame):
    # 81x72
    frame = frame[32:193, 8:152]  # Crop the image
    frame = frame[::2, ::2]  # Reduce dimensions by taking every 2nd pixel

    frame = np.mean(frame, axis=2).astype(np.uint8)  # Grayscale - change (R,G,B) to (COLOUR)
    frame = frame / 255.0  # Normalize pixel values

    return frame

def stack_frames(frame_deque, frame, is_new_episode):
    if is_new_episode:
        # Clear our frame_deque
        frame_deque = deque(maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        frame_deque.append(frame)
        frame_deque.append(frame)
        frame_deque.append(frame)
        frame_deque.append(frame)

        # Stack the frames
        stacked_state = np.stack(frame_deque, axis=0)

    else:
        # Append frame to deque, automatically removes the oldest frame
        frame_deque.append(frame)
        stacked_state = np.stack(frame_deque, axis=0)

    return stacked_state, frame_deque



class DQN:
    def __init__(self, n_outputs, conv_layer_sizes, hidden_layer_sizes, gamma, scope, max_experiences, min_experiences,
                 batch_size):
        self.n_outputs = n_outputs
        self.scope = scope

        with tf.variable_scope(scope):
            self.X = tf.placeholder(tf.float32, shape=(None, 4, HEIGHT, WIDTH), name='X')
            self.G = tf.placeholder(tf.float32, (None,), name='G')
            self.actions = tf.placeholder(tf.int32, (None, ), "actions")

        Z = self.X
        Z = tf.transpose(Z, [0, 2,3,1])

        for num_output_filters, filtersz, poolsz in conv_layer_sizes:
            Z = tf.contrib.layers.conv2d(
                Z,
                num_output_filters,
                filtersz,
                poolsz,
                activation_fn = tf.nn.relu
            )

        Z = tf.contrib.layers.flatten(Z)
        for M in hidden_layer_sizes:
            Z = tf.contrib.layers.fully_connected(Z, M)

        self.predict_op = tf.contrib.layers.fully_connected(Z, n_outputs)

        selected_action_values = tf.reduce_sum(
            self.predict_op * tf.one_hot(self.actions, n_outputs),
            reduction_indices=[1]
        )

        cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
        self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(cost)

        self.cost = cost

        # Create replay memory
        self.experience = []
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.gamma = gamma

    def set_session(self, session):
        self.session = session

    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)

        ops = []
        for p, q in zip(mine, theirs):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)

        self.session.run(ops)


    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def update(self, states, actions, targets):
        cost, _ = self.session.run(
            [self.cost, self.train_op],
            feed_dict= {
                self.X : states,
                self.G : targets,
                self.actions : actions
            }
        )

        return cost

    def sample_action(self, state, eps):
        if np.random.random() < eps:
            return np.random.choice(self.n_outputs)
        else:
            return np.argmax(self.predict([state])[0])

def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
    samples = random.sample(experience_replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

    next_Qs = target_model.predict(next_states)
    next_Q = np.amax(next_Qs, axis=1)

    targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q

    loss = model.update(states, actions, targets)
    return loss


def play_one(env,
             total_t,
             experience_replay_buffer,
             model,
             target_model,
             gamma,
             batch_size,
             epsilon,
             epsilon_change,
             epsilon_min):

    t0 = datetime.now()
    raw_frame = env.reset()
    frame = preprocess_frame(raw_frame)
    state, frame_deque = stack_frames(None, frame, True)
    assert(state.shape == (4, 81, 72))
    loss = None

    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0

    done = False
    while not done:

        if total_t % TARGET_UPDATE_PERIOD == 0:
            target_model.copy_from(model)
            print("Copied model parameters to target network. total_t = %s, period = %s" % (
            total_t, TARGET_UPDATE_PERIOD))

        action = model.sample_action(state, epsilon)
        raw_frame, reward, done, _ = env.step(action)
        frame = preprocess_frame(raw_frame)

        next_state = np.append(state[1:], np.expand_dims(frame, 0), axis=0)
        assert (state.shape == (4, 81, 72))

        episode_reward+=reward

        if len(experience_replay_buffer) == MAX_EXPERIENCES:
            experience_replay_buffer.pop(0)

        experience_replay_buffer.append((state, action, reward, next_state, done))

        t0_2 = datetime.now()
        loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
        dt = datetime.now() - t0_2

        total_time_training+= dt.total_seconds()
        num_steps_in_episode+= 1

        state = next_state
        total_t+=1

        epsilon = max(epsilon - epsilon_change, epsilon_min)

    return total_t, episode_reward, (
                datetime.now() - t0), num_steps_in_episode, total_time_training / num_steps_in_episode, epsilon

if __name__ == '__main__':

    conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    hidden_layer_sizes = [512]

    gamma = 0.99
    batch_size = 32
    num_episodes = 10000
    total_t = 0
    experience_replay_buffer = []

    episode_rewards = np.zeros(num_episodes)

    epsilon = 1.0
    epsilon_min = 0.1

    epsilon_change = (epsilon-epsilon_min) / MAX_EXPERIENCES

    env = gym.make('Breakout-v0')
    n_actions = env.action_space.n


    model = DQN(
        env.action_space.n,
        conv_layer_sizes,
        hidden_layer_sizes,
        gamma,
        "model",
        MAX_EXPERIENCES,
        MIN_EXPERIENCES,
        batch_size
    )

    target_model = DQN(
        env.action_space.n,
        conv_layer_sizes,
        hidden_layer_sizes,
        gamma,
        "target_model",
        MAX_EXPERIENCES,
        MIN_EXPERIENCES,
        batch_size
    )

    with tf.Session() as sess:
        model.set_session(sess)
        target_model.set_session(sess)

        sess.run(tf.global_variables_initializer())

        print("Populating experience replay buffer")
        raw_frame = env.reset()
        frame = preprocess_frame(raw_frame)

        state, frame_deque = stack_frames(None, frame, True)

        assert(state.shape == (4, 81, 72))

        for i in range(MIN_EXPERIENCES):

            action = np.random.choice(n_actions)
            raw_frame, reward, done, _ = env.step(action)
            frame = preprocess_frame(raw_frame)
            next_state, frame_deque = stack_frames(frame_deque, frame, False)
            assert (state.shape == (4, 81, 72))
            experience_replay_buffer.append((state, action, reward, next_state, done))

            if done:
                raw_frame = env.reset()
                frame = preprocess_frame(raw_frame)
                state, frame_deque = stack_frames(None, frame, True)
            else:
                state = next_state

        print("STARTING TRAINING for {} episodes".format(num_episodes))
        for i in range(num_episodes):
            total_t, episode_reward, duration, num_steps_in_ep, time_per_step, epsilon = play_one(
                env,
                total_t,
                experience_replay_buffer,
                model,
                target_model,
                gamma,
                batch_size,
                epsilon,
                epsilon_change,
                epsilon_min
            )
            episode_rewards[i] = episode_reward

            last_100_avg = episode_rewards[max(0, i - 100): i + 1].mean()
            print("Episode:", i,
                  "Duration:", duration,
                  "Num steps:", num_steps_in_ep,
                  "Reward:", episode_reward,
                  "Training time per step:", "%.3f" % time_per_step,
                  "Avg Reward (Last 100):", "%.3f" % last_100_avg,
                  "Epsilon:", "%.3f" % epsilon
                  )
            sys.stdout.flush()

