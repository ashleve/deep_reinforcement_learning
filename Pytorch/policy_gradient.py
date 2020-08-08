import gym
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import matplotlib.pyplot as plt


env_name = 'CartPole-v0'
env = gym.make(env_name)

obs_space = env.observation_space.shape[0]
action_space = env.action_space.n

model = nn.Sequential(
    nn.Linear(obs_space, 10),
    nn.Linear(10, 10),
    nn.Linear(10, action_space)
)
optimizer = Adam(model.parameters(), lr=0.01)


def get_policy(obs):
    logits = model(obs)
    return Categorical(logits=logits)


def get_action(obs):
    return get_policy(obs).sample().item()


def compute_loss(states, actions, returns):
    return -(get_policy(states).log_prob(actions) * returns).mean()


def policy_update(states, actions, returns):
    model.zero_grad()
    loss = compute_loss(
        torch.as_tensor(states, dtype=torch.float32),
        torch.as_tensor(actions, dtype=torch.int32),
        torch.as_tensor(returns, dtype=torch.float32),
    )
    loss.backward()
    optimizer.step()


def calculate_returns(rewards):
    returns = []
    for i in range(len(rewards)):
        returns.append(sum(rewards[i:]))
    return returns


def play_one_game(max_steps=1000):
    s = env.reset()

    state_hist = []
    action_hist = []
    reward_hist = []

    for _ in range(max_steps):
        a = get_action(torch.as_tensor(s, dtype=torch.float32))

        # unlock to compare with random algorithm
        # a = env.action_space.sample()

        new_s, r, done, info = env.step(a)

        # env.render()

        state_hist.append(s)
        action_hist.append(a)
        reward_hist.append(r)

        if done:
            break

        s = new_s

    total_reward = sum(reward_hist)
    return_hist = calculate_returns(reward_hist)
    return state_hist, action_hist, return_hist, total_reward


def train_one_epoch(batch_size=128):

    states = []
    actions = []
    returns = []

    mean_reward = 0
    i = 0

    while len(states) < batch_size:
        state_hist, action_hist, return_hist, total_reward = play_one_game()
        states += state_hist
        actions += action_hist
        returns += return_hist
        mean_reward += total_reward
        i += 1

    mean_reward /= i

    policy_update(states, actions, returns)

    return mean_reward


def train(num_of_epochs=250):
    rewards = []
    for i in range(num_of_epochs):
        mean_reward = train_one_epoch()
        print(f"Epoch {i}, mean reward: {mean_reward}")
        rewards.append(mean_reward)
    return rewards


def main():
    rewards = train()
    plt.plot(rewards)
    plt.show()


if __name__ == "__main__":
    main()
