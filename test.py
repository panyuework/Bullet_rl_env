from Env.envs.x10car_env_out20 import X10Car_Env_out20
from Env.envs.x10car_env_out50 import X10Car_Env_out50

env = X10Car_Env_out20()
# env = X10Car_Env_out50()


episode = 0
agent_steps = 0
episode_reward = 0
max_steps = 100000000  # maximum steps to evaluate

n_steps = max_steps
for step in range(n_steps):

    action = [0.4, 0.36]  # select action using the trained model
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, reward, done, info = env.step(action)  # perform action to obtain new observation and reward
    agent_steps = agent_steps + 1
    episode_reward = episode_reward + reward
    print('obs=', obs, 'reward=', reward, 'done=', done)
    print('info', info)
    # env.render()

    if done:
        episode = episode + 1
        episode_reward = episode_reward + reward
        obs = env.reset()
        print("Obstacle Hit,", "Episode =", episode, ",", "Agent Steps =", agent_steps, ",", "Episode Reward =",
              episode_reward)
        agent_steps = 0
        episode_reward = 0