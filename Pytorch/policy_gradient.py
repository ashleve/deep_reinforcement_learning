import gym
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import matplotlib.pyplot as plt


env_name = 'CartPole-v0'
# env_name = 'LunarLander-v2'
env = gym.make(env_name)
obs_space = env.observation_space.shape[0]
action_space = env.action_space.n


model = nn.Sequential(
    nn.Linear(obs_space, 32),
    nn.ReLU(),
    nn.Linear(32, action_space)
)
optimizer = Adam(model.parameters(), lr=0.01)


def get_policy(state):
    logits = model(state)
    return Categorical(logits=logits)


def get_action(state):
    state = torch.as_tensor(state, dtype=torch.float32)
    return get_policy(state).sample().item()


def compute_loss(states, actions, rewards_to_go):
    return -(get_policy(states).log_prob(actions) * rewards_to_go).mean()


def policy_update(states, actions, rewards):
    rewards_to_go = compute_r2g(rewards, gamma=0.99)

    model.zero_grad()
    loss = compute_loss(
        torch.as_tensor(states, dtype=torch.float32),
        torch.as_tensor(actions, dtype=torch.int32),
        torch.as_tensor(rewards_to_go, dtype=torch.float32),
    )
    loss.backward()
    optimizer.step()


def compute_r2g(rewards, gamma):
    """
        Calculates discounted rewards-to-go and normalizes them.
        Params:
            gamma - Discount.
    """

    rewards2go = []
    running_sum = 0
    for r in rewards[::-1]:
        running_sum = r + gamma * running_sum
        rewards2go.insert(0, running_sum)

    # normalize rewards
    rewards2go = torch.tensor(rewards2go)
    rewards_normalized = (rewards2go - rewards2go.mean()) / (rewards2go.std() + 0.0001)

    return rewards_normalized


def play_one_game(max_steps=10000, render=False):
    s = env.reset()
    states = []
    actions = []
    rewards = []

    for _ in range(max_steps):
        a = get_action(s)
        new_s, r, done, info = env.step(a)

        if render:
            env.render()

        states.append(s)
        actions.append(a)
        rewards.append(r)

        if done:
            break

        s = new_s

    return states, actions, rewards


def train(num_of_episodes=250):
    reward_hist = []

    for i in range(num_of_episodes):

        states, actions, rewards = play_one_game()

        policy_update(states, actions, rewards)

        total_reward = sum(rewards)
        print(f"Episode {i}, length: {len(states)} reward: {total_reward}")
        reward_hist.append(total_reward)

    return reward_hist


def main():
    reward_hist = train()
    plt.plot(reward_hist)
    plt.show()


if __name__ == "__main__":
    main()
