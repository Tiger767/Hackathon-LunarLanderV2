"""
Authors: Travis
"""

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


def dense(units, activation='relu', l1=0, l2=0, batch_norm=True,
          momentum=0.999, epsilon=1e-5, name=None):
    """Creates a dense layer function.
    params:
        units: An integer, which is the dimensionality of the output space
        activation: A string or keras/TF activation function
        l1: A float, which is the amount of L1 regularization
        l2: A float, which is the amount of L2 regularization
        batch_norm: A boolean, which determines if batch
                    normalization is enabled
        momentum: A float, which is the momentum for the moving
                  mean and variance
        epsilon: A float, which adds variance to avoid dividing by zero
        name: A string, which is the name of the dense layer
    return: A function, which takes a layer as input and returns a dense(layer)
    """
    if activation == 'relu':
        kernel_initializer = 'he_normal'
    else:
        kernel_initializer = 'glorot_uniform'
    if l1 == l2 == 0:
        dl = keras.layers.Dense(units, activation=activation, name=name,
                                kernel_initializer=kernel_initializer,
                                use_bias=not batch_norm)
    else:
        dl = keras.layers.Dense(units, activation=activation,
                                kernel_regularizer=l1_l2(l1, l2), name=name,
                                kernel_initializer=kernel_initializer,
                                use_bias=not batch_norm)
    if batch_norm:
        bn_name = name + '_batchnorm' if name is not None else None
        bnl = keras.layers.BatchNormalization(epsilon=epsilon,
                                              momentum=momentum,
                                              name=bn_name)

    def layer(x):
        """Applies dense layer to input layer.
        params:
            x: A Tensor
        return: A Tensor
        """
        x = dl(x)
        if batch_norm:
            x = bnl(x)
        return x
    return layer


def create_amodel(state_shape, action_shape):
    inputs = keras.layers.Input(shape=state_shape)
    x = dense(32)(inputs)
    x = dense(32)(x)
    outputs = dense(action_shape[0], activation='softmax',
                    batch_norm=False)(x)
    
    amodel = keras.Model(inputs=inputs,
                         outputs=outputs)
    amodel.compile(optimizer=keras.optimizers.Adam(.003),
                   loss='mse', experimental_run_tf_function=False)
    amodel.summary()
    return amodel


def create_cmodel(state_shape):
    inputs = keras.layers.Input(shape=state_shape)
    x = dense(64)(inputs)
    x = dense(64)(x)
    outputs = keras.layers.Dense(1)(x)

    cmodel = keras.Model(inputs=inputs,
                         outputs=outputs)
    cmodel.compile(optimizer=keras.optimizers.Adam(.001),
                   loss='mse', experimental_run_tf_function=False)
    cmodel.summary()
    return cmodel


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

    agents_to_use = ['DQN', 'PG', 'A2C']
    agent_to_use = agents_to_use[2]

    if agent_to_use == 'DQN':
        pass
    elif agent_to_use == 'PG':
        amodel = create_amodel(env.state_shape, env.action_shape)
        agent = PGAgent(amodel, .999, create_memory=lambda: RingMemory(500000))

        # Uncomment if you want to play random exploring episodes
        agent.set_playing_data(training=False, memorizing=True)
        env.play_episodes(agent, 100, max_steps, random=True,
                          verbose=True, episode_verbose=False,
                          render=False)
        agent.save(save_dir, note=f'PG')
        print(len(agent.states))
        #agent.learn(batch_size=32, epochs=10)
        #agent.load(f'{save_dir}/20200411_120627_345100', load_data=False)
        #agent.amodel.compile(optimizer=keras.optimizers.Adam(.003),
        #                     loss='mse', experimental_run_tf_function=False)

        # Train
        agent.set_playing_data(training=True, memorizing=True,
                                batch_size=32, mini_batch=10000,
                                entropy_coef=0,
                                verbose=True)
        for ndx in range(50):
            print(f'Save Loop: {ndx}')
            env.play_episodes(agent, 1, max_steps,
                                verbose=True, episode_verbose=False,
                                render=True)
            result = env.play_episodes(agent, 19, max_steps,
                                       verbose=True, episode_verbose=False,
                                       render=False)
            agent.save(save_dir, note=f'PG_{ndx}_{result}')
            if result >= solved:
                break

        # Test
        agent.set_playing_data(training=False, memorizing=False)
        env.play_episodes(agent, 1, max_steps,
                          verbose=True, episode_verbose=False,
                          render=True)
        avg = env.play_episodes(agent, 100, max_steps,
                                verbose=True, episode_verbose=False,
                                render=False)
        print(len(agent.states))
        print(avg)
    elif agent_to_use == 'A2C':
        amodel = create_amodel(env.state_shape, env.action_shape)
        cmodel = create_cmodel(env.state_shape)
        agent = A2CAgent(amodel, cmodel, .99, #lambda_rate=0.95,
                         create_memory=lambda: RingMemory(500000))

        agent.set_playing_data(training=True, memorizing=True,
                               batch_size=64, mini_batch=10000, epochs=1,
                               entropy_coef=0,
                               verbose=True)
        for ndx in range(50):
            print(f'Save Loop: {ndx}')
            env.play_episodes(agent, 1, max_steps,
                                verbose=True, episode_verbose=False,
                                render=True)
            result = env.play_episodes(agent, 19, max_steps,
                                       verbose=True, episode_verbose=False,
                                       render=False)
            agent.save(save_dir, note=f'A2C_{ndx}_{result}')
            if result >= solved:
                break

        # Test
        agent.set_playing_data(training=False, memorizing=False)
        env.play_episodes(agent, 1, max_steps,
                          verbose=True, episode_verbose=False,
                          render=True)
        avg = env.play_episodes(agent, 100, max_steps,
                                verbose=True, episode_verbose=False,
                                render=False)
        print(len(agent.states))
        print(avg)