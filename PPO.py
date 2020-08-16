import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
import matplotlib.pyplot as plt


# env_name = 'CartPole-v0'
env_name = 'LunarLander-v2'
env = gym.make(env_name)
obs_space = env.observation_space.shape[0]
action_space = env.action_space.n
torch.manual_seed(123)


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add(self, state, action):
        self.states.append(state)
        self.actions.append(action)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Linear(obs_space, 64),
            nn.Tanh()
        )
        self.action_layer = nn.Sequential(
            nn.Linear(64, action_space),
            nn.Softmax()
        )
        self.value_layer = nn.Linear(64, 1)

    def forward(self, state):
        x = self.first_layer(state)
        pi = self.action_layer(x)
        v = self.value_layer(x)
        return Categorical(pi), torch.squeeze(v)


actor_critic = ActorCritic()
optimizer = Adam(actor_critic.parameters(), lr=0.02)
memory = Memory()


def compute_r2g(rewards, gamma):
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

    return rewards2go


def update():
    optimizer.zero_grad()

    states = torch.as_tensor(memory.states, dtype=torch.float32)
    actions = torch.as_tensor(memory.actions, dtype=torch.int32)
    rewards = torch.as_tensor(memory.rewards, dtype=torch.float32)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

    pi, v = actor_critic(states)
    old_log_probs = pi.log_prob(actions).detach()   # detach() removes gradient
    adv = (rewards - v).detach()

    epochs = 5
    eps = 0.2
    for _ in range(epochs):
        pi, v = actor_critic(states)
        log_probs = pi.log_prob(actions)

        # exp(log_prob - old_log_prob) is the same as (prob / old_prob)
        ratio = torch.exp(log_probs - old_log_probs)
        clip = torch.clamp(ratio, 1-eps, 1+eps)
        loss = -torch.min(ratio * adv, clip * adv).mean() + F.smooth_l1_loss(rewards, v)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def play_one_game(max_steps=10000):
    s = env.reset()
    rewards = []

    for i in range(max_steps):
        pi, v = actor_critic(torch.as_tensor(s, dtype=torch.float32))
        a = pi.sample().item()

        new_s, r, done, info = env.step(a)

        rewards.append(r)
        memory.add(state=s, action=a)

        if done:
            break

        s = new_s

    rewards_to_go = compute_r2g(rewards, gamma=0.98)
    memory.rewards.extend(rewards_to_go)

    return sum(rewards), i


def train(num_of_games=1000):
    reward_hist = []

    for i in range(1, num_of_games):
        total_reward, ep_length = play_one_game()
        reward_hist.append(total_reward)

        if i % 5 == 0:
            update()
            memory.clear()

        if i % 20 == 0:
            print(f"Episode {i}, length: {ep_length}, reward: {total_reward}")

    return reward_hist


def main():
    reward_hist = train(num_of_games=1500)
    env.close()

    plt.plot(reward_hist)
    plt.show()


if __name__ == "__main__":
    main()


