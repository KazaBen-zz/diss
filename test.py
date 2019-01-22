import gym
import time

env = gym.make('Breakout-v0')
frame = env.reset()
print(frame.shape)
env = gym.make('CartPole-v0')
frame = env.reset()
print(frame.shape)

# while True:
#     env.step(1)
#     env.render()
#     time.sleep(0.3)
