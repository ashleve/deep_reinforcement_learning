import gym
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import matplotlib.pyplot as plt


# env_name = 'CartPole-v0'
env_name = 'LunarLander-v2'

env = gym.make(env_name)

obs_space = env.observation_space.shape[0]
action_space = env.action_space.n


class Actor:
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(obs_space, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )
        self.optimizer = Adam(self.net.parameters(), lr=0.01)

    def get_policy(self, state):
        logits = self.net(state)
        return Categorical(logits=logits)

    def get_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        return self.get_policy(state).sample().item()

    def compute_loss(self, states, actions, adv):
        return -(self.get_policy(states).log_prob(actions) * adv).mean()


class Critic:
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(obs_space, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = Adam(self.net.parameters(), lr=0.01)

    def get_state_values(self, states):
        states = torch.as_tensor(states, dtype=torch.float32)
        return self.net(states)

    def compute_loss(self, state_values, rewards):
        return (state_values - rewards).pow(2).mean()


actor = Actor()
critic = Critic()


def r2g(rewards, gamma):
    """
        Computes discounted rewards-to-go and normalizes them.
        Params:
            gamma - Discount.
    """
    rewards2go = []
    running_sum = 0
    for r in rewards[::-1]:
        running_sum = r + gamma * running_sum
        rewards2go.insert(0, running_sum)

    rewards2go = torch.tensor(rewards2go)
    rewards_normalized = (rewards2go - rewards2go.mean()) / rewards2go.std()

    return rewards_normalized


def compute_advantage_estimations(rewards, state_values):
    return [r - state_values[i] for i, r in enumerate(rewards)]


def update_actor(states, actions, advs):
    actor.net.zero_grad()
    loss = actor.compute_loss(states, actions, advs)
    print(loss)
    loss.backward()
    actor.optimizer.step()


def update_critic(state_values, returns):
    critic.net.zero_grad()
    loss = critic.compute_loss(state_values, returns)
    print(loss)
    loss.backward()
    critic.optimizer.step()


def update(states, actions, rewards):
    state_values = torch.squeeze(critic.get_state_values(states), -1)
    advs = compute_advantage_estimations(rewards, state_values)

    states = torch.as_tensor(states, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.int32)
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    advs = torch.as_tensor(advs, dtype=torch.float32)

    update_actor(states, actions, advs)
    update_critic(state_values, rewards)


def play_one_game(max_steps=10000):
    s = env.reset()
    state_hist = []
    action_hist = []
    reward_hist = []
    for _ in range(max_steps):
        a = actor.get_action(s)

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
    discount = 0.95
    rewards_to_go = r2g(reward_hist, discount)
    return state_hist, action_hist, rewards_to_go, total_reward


def train_one_epoch(batch_size=256):
    states = []
    actions = []
    rewards = torch.tensor([])

    mean_reward = 0
    i = 0

    while len(states) < batch_size:
        state_hist, action_hist, rewards_to_go, total_reward = play_one_game(max_steps=200)
        states += state_hist
        actions += action_hist
        rewards = torch.cat((rewards, rewards_to_go), 0)
        mean_reward += total_reward
        i += 1

    mean_reward /= i
    update(states, actions, rewards)
    return mean_reward


def train(num_of_epochs=250):
    rewards = []
    for i in range(num_of_epochs):
        mean_reward = train_one_epoch(batch_size=256)
        print(f"Epoch {i}, mean reward: {mean_reward}")
        rewards.append(mean_reward)
    return rewards


def main():
    rewards = train(num_of_epochs=500)

    plt.plot(rewards)
    plt.show()

    for _ in range(10):
        s = env.reset()
        while True:
            a = actor.get_action(torch.as_tensor(s, dtype=torch.float32))
            s, r, done, info = env.step(a)
            env.render()
            if done:
                break


if __name__ == "__main__":
    main()
