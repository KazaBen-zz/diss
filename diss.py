import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf

# np.set_printoptions(threshold= np.nan)


# Actions:
# 1 - dont move
# 2 - go right
# 3 - go left

ENVIRONMENT = 'Breakout-v0'
env = gym.make(ENVIRONMENT)
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())

# MODEL HYPERPARAMETERS
STATE_SIZE = [81, 72, 4]
ACTION_SIZE = env.action_space.n
ALPHA = 0.00025

# TRAINING HYPERPARAMETERS
TOTAL_EPISODES = 100000
MAX_STEPS = 50000
BATCH_SIZE = 64

# EPSILON GREEDY HYPERPARAMETERS
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
DECAY_RATE = 0.00001

# Q-LEARNING HYPERPARAMETERS
GAMMA = 0.97

stack_size = 4
training = True
episode_render = False

# MEMORY HYPERPARAMETERS
PRETRAIN_LENGTH = BATCH_SIZE
MEMORY_SIZE = 1000000


def preprocess_frame(frame):
    # 81x72
    frame = frame[32:193, 8:152] # Crop the image
    frame = frame[::2, ::2] # Reduce dimensions by taking every 2nd pixel

    frame = np.mean(frame, axis=2).astype(np.uint8) # Grayscale - change (R,G,B) to (COLOUR)
    frame = frame/255.0 # Normalize pixel values

    return frame


def stack_frames(frame_deque, new_frame, is_new_episode):
    # TODO: Frame skipping = 2
    frame = preprocess_frame(new_frame)
    if is_new_episode:
        # Clear our frame_deque
        frame_deque = deque([np.zeros((81, 72), dtype=np.int) for _ in range(4)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        frame_deque.append(frame)
        frame_deque.append(frame)
        frame_deque.append(frame)
        frame_deque.append(frame)

        # Stack the frames
        stacked_state = np.stack(frame_deque, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        frame_deque.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(frame_deque, axis=2)

    return stacked_state, frame_deque



class DQN:
    def __init__(self, state_size, action_size, learning_rate, name='DQN'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # Create placeholders
            self.inputs = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions = tf.placeholder(tf.float32, [None, self.action_size], name="actions")

            # target_Q = R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.conv1 = tf.layers.conv2d(inputs = self.inputs,
                                          filters = 32,
                                          kernel_size = [8,8],
                                          padding = "VALID",
                                          # kerner_initalizer = tf.contrib.layers.xavier_initializer_conv2d(),
                                          name = "conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                          filters = 64,
                                          kernel_size=[4,4],
                                          strides=[2,2],
                                          padding="VALID",
                                          # kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                          filters = 64,
                                          kernel_size = [3,3],
                                          strides = [2,2],
                                          padding = "VALID",
                                          # kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units = 512,
                                      activation = tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name = "fc1")

            self.output = tf.layers.dense(inputs = self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size,
                                          activation=None)

            # Predicted Q value
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions))

            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen= max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size = batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


tf.reset_default_graph()
DQN = DQN(STATE_SIZE, ACTION_SIZE, ALPHA)
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())

# Initiate memory
memory = Memory(max_size=MEMORY_SIZE)
for i in range(PRETRAIN_LENGTH):
    if i == 0:
        raw_frame = env.reset()
        state, frame_deque = stack_frames(None, raw_frame, True)

    choice = np.random.randint(0, len(possible_actions))
    action = possible_actions[choice]
    env.step(1) # 1 frame skip TODO
    next_raw_frame, reward, done, _ = env.step(choice)

    next_state, frame_deque = stack_frames(frame_deque, next_raw_frame, False)

    if done:
        next_state = np.zeros(state.shape)

        memory.add((state, action, reward, next_state, done))
        raw_frame = env.reset()
        state, frame_deque = stack_frames(frame_deque, raw_frame, True)
    else:
        memory.add((state, action, reward, next_state, done))
        state = next_state




writer = tf.summary.FileWriter("tensorboard/dqn/1")
tf.summary.scalar("Loss", DQN.loss)
write_op = tf.summary.merge_all()

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, action):
    # EPS GREEDY
    # p = np.random.rand()
    p = 1
    explore_prob = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if explore_prob > p:
        # Random action(Exploration)
        choice = np.random.randint(0, len(possible_actions))
    else:
        # Exploit
        Qs = sess.run(DQN.output, feed_dict = {DQN.inputs : state.reshape((1, *state.shape))})

        choice = np.argmax(Qs)
    return choice, explore_prob

saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rewards_list = []
        decay_step = 0

        for episode in range(TOTAL_EPISODES):
            step = 0
            episode_rewards = []
            frame = env.reset()

            stacked_state, frame_deque = stack_frames(None, frame, True)

            while step < MAX_STEPS:
                step += 1
                decay_step += 1
                choice, explore_probability = predict_action(EPSILON_MAX, EPSILON_MIN, DECAY_RATE, decay_step, stacked_state, possible_actions)
                action = possible_actions[choice]
                env.step(1)  # 1 frame skip TODO
                next_state, reward, done, _ = env.step(choice)

                if episode_render:
                    env.render()

                episode_rewards.append(reward)

                if done:
                    next_state = np.zeros((210, 160, 3), dtype=np.int) # TODO maybe not stack?

                    next_state, frame_deque = stack_frames(frame_deque, next_state, False)

                    step = MAX_STEPS # to end ep

                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Explore P: {:.4f}'.format(explore_probability),
                          'Training Loss: {:.4f}'.format(loss))

                    rewards_list.append((episode, total_reward))

                    memory.add((state, action, reward, next_state, done))

                else:
                    next_state, frame_deque = stack_frames(frame_deque, next_state, False)

                    memory.add((state, action, reward, next_state, done))

                    state = next_state


                batch = memory.sample(BATCH_SIZE)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []
                Qs_next_state = sess.run(DQN.output, feed_dict={DQN.inputs: next_states_mb}) # get Qs for s'
                   
                if(episode == 0 and step == 1):
                    print("RESTORING SAVED MODEL")
                    saver.restore(sess, "models/model.ckpt")



                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + GAMMA * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)

                target_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([DQN.loss, DQN.optimizer],
                                   feed_dict= {
                                       DQN.inputs :states_mb,
                                       DQN.target_Q : target_mb,
                                       DQN.actions : actions_mb})

                summary = sess.run(write_op, feed_dict={
                    DQN.inputs : states_mb,
                    DQN.target_Q : target_mb,
                    DQN.actions : actions_mb
                })

                writer.add_summary(summary, episode)
                writer.flush()

            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model saved")
