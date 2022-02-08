import gym

# get the Pong-v0 environment instance
env = gym.make('Pong-v0', render_mode='human')

# instantiate the wrapper used for monitoring, the recoding will be saved in '../recording/basics/pong/' directory
env = gym.wrappers.Monitor(env, '../recordings/basics/pong/', force=True)

# you need reset the environment before you begin to interact with
env.reset()

# let's take random actions for 1000 times
for _ in range(1000):
    # random action is selected from the action space
    action = env.action_space.sample()

    # return values for the action performed on the environment
    next_state, reward, is_terminal_state, info = env.step(action)
    if is_terminal_state:
        break
env.close()
