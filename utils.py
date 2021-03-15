import os
from gym.envs.registration import register
import numpy as np
def registerEnvs(env_name, max_episode_steps, xml_dir):
    """register the MuJoCo envs with Gym and return the per-limb observation size and max action value (for modular policy training)"""

    # env_name = os.path.basename(xml_dir)[:-4]
    # env_file = env_name
    # create a copy of modular environment for custom xml model
    params = {'xml': os.path.abspath(xml_dir)}
    # register with gym
    register(id=("%s-v0" % env_name),
                max_episode_steps=max_episode_steps,
                entry_point="ant_maze_env:AntMazeEnv")
    print("finish registration")