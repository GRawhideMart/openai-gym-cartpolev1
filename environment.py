import gymnasium as gym

env = gym.make('CartPole-v1', render_mode = 'human')
state, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
        state, info = env.reset()
    env.render()

env.close()