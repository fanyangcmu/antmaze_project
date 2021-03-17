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
            goal_1_suc = info['goal_1_flag']
            goal_2_suc = info['goal_2_flag']
            goal_3_suc = info['goal_3_flag']
