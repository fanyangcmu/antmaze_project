import gym
from utils import registerEnvs
MAX_EPISODE_TIMESTEPS = 1500


if __name__ == "__main__":
    registerEnvs("ant_maze", MAX_EPISODE_TIMESTEPS, 'assets/ant.xml')
    env = gym.make("ant_maze-v0")
    obs = env.reset()
    done = False
    episodes = 10
    for i in range(episodes):
        while not done:
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)