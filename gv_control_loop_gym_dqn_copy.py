#!/usr/bin/env python
# not tested!!!
import argparse
import itertools as itt
import time
from typing import Dict

import pandas as pd
import gym
import numpy as np
import torch as ch
from policy_DQN import *

from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.gym import GymEnvironment
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    make_observation_representation,
)
from gym_gridverse.representations.state_representations import (
    make_state_representation,
)


def make_env(id_or_path: str) -> GymEnvironment:
    """Makes a GV gym environment."""
    try:
        print('Loading using gym.make')
        env = gym.make(id_or_path)

    except gym.error.Error:
        print(f'Environment with id {id_or_path} not found.')
        print('Loading using YAML')
        inner_env = factory_env_from_yaml(id_or_path)
        state_representation = make_state_representation(
            'compact',
            inner_env.state_space,
        )
        observation_representation = make_observation_representation(
            'compact',
            inner_env.observation_space,
        )

        outer_env = OuterEnv(
            inner_env,
            state_representation=state_representation,
            observation_representation=observation_representation,
        )
        env = GymEnvironment(outer_env)

    else:
        if not isinstance(env, GymEnvironment):
            raise ValueError(
                f'gym id {id_or_path} is not associated with a GridVerse environment'
            )

    return env


def print_compact(data: Dict[str, np.ndarray]):
    """Converts numpy arrays into lists before printing, for more compact output."""
    if isinstance(data, dict):
        return np.concatenate([v.flatten() for v in data.values()])
    elif isinstance(data, np.ndarray):
        return data.flatten()
    else:
        raise TypeError("Unsupported observation type")


def main(args):
    env = make_env(args.id_or_path)
    observation = env.reset()
    observation_space = preprocess_observation(observation)
    num_states = observation_space.shape[0]
    network = get_network(num_states, env.action_space.n)
    opt = torch.optim.Adam(network.parameters(), lr=1e-4)

    # load model if exists, AND IF args.load_model is True
    if args.model:
        try:
            network.load_state_dict(ch.load(args.model))
        except FileNotFoundError:
            pass

    spf = 1 / args.fps

    total_reward_list = []

    for ei in range(200, 300):
        print(f'# Episode {ei}')
        # print()

        total_reward = 0
        observation = preprocess_observation(env.reset())

        # what is the actual difference of observation and state, in code?

        # env.render()

        # print('observation:')
        # print_compact(observation)
        # print()

        # time.sleep(spf)
        for ti in itt.count():
            # print(f'episode: {ei}')
            # print(f'time: {ti}')

            # action = env.action_space.sample()
            action = get_action(observation, network)
            observation, reward, done, _ = env.step(action)

            opt.zero_grad()
            loss = compute_td_loss(
                observation, action, reward, observation, done, network
            )
            loss.backward()
            opt.step()

            total_reward += reward

            # env.render()

            # print(f'total reward: {total_reward}')
            # print(f'action: {action}')
            # print(f'reward: {reward}')
            # print('observation:')
            # print_compact(observation)
            # print(f'done: {done}')
            # print()

            # time.sleep(spf)

            if done:
                print(f'time: {ti}')
                print(f'total reward: {total_reward}')
                break

        total_reward_list.append(total_reward)

    df = pd.DataFrame(total_reward_list, columns=['Total Reward'])
    df.to_csv('total_rewards.csv', index_label='Episode')
    ch.save(network.state_dict(), 'model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('id_or_path', help='Gym id or GV YAML file')
    parser.add_argument(
        '--fps', type=float, default=1.0, help='frames per second'
    )
    parser.add_argument(
        '--model', default=None, help='load model if path is given'
    )
    main(parser.parse_args())
