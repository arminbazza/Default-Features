import gymnasium as gym
from minigrid.envs import EmptyEnv
from minigrid.envs.fourrooms import FourRoomsEnv

from tabular_wrapper import TabularWrapper

env = gym.make('MiniGrid-Empty-16x16-v0')
env = TabularWrapper(env)