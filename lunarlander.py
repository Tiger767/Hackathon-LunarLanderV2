import gym 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.reinforcement import (
    PGAgent, DQNAgent, StochasticPolicy, GreedyPolicy,
    ExponentialDecay, RingMemory, Memory, GymWrapper
)
from utils.reinforcement_agents import (
    A2CAgent, PPOAgent
)

if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True   
    sess = tf.compat.v1.Session(config=config)

    # Solved = 200 avg. reward over 100 episodes

    solved = 200
    save_dir = 'lunarlander_saves'
    env = gym.make('LunarLander-v2')
    max_steps = env._max_episode_steps  # (1000)
    env = GymWrapper(env, (8,), (4,))

    print('close')
