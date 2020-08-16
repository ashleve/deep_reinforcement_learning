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
        self.rewards = []
        self.log_probs = []
        self.state_values = []

    def add(self, reward, log_prob, state_value):
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)

    def clear(self):
        self.rewards.clear()
        self.log_probs.clear()
        self.state_values.clear()


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Linear(obs_space, 64),
            nn.ReLU()
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

    def compute_loss(self, rewards_to_go, log_probs, state_values):
        adv = rewards_to_go - state_values
        adv = adv.detach()  # remove gradient from advantage estimations

        action_loss = -(log_probs * adv).mean()
        # value_loss = F.mse_loss(state_values, rewards_to_go)
        value_loss = F.smooth_l1_loss(state_values, rewards_to_go)

        return action_loss + value_loss


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

    # normalize rewards
    rewards2go = torch.tensor(rewards2go)
    rewards_normalized = (rewards2go - rewards2go.mean()) / (rewards2go.std() + 0.00001)

    return rewards_normalized


def update():
    rewards_to_go = compute_r2g(memory.rewards, gamma=0.97)

    rewards_to_go = torch.as_tensor(rewards_to_go, dtype=torch.float32)
    log_probs = torch.stack(memory.log_probs)  # transform list of tensors to single tensor
    state_values = torch.stack(memory.state_values)

    optimizer.zero_grad()
    loss = actor_critic.compute_loss(rewards_to_go, log_probs, state_values)
    loss.backward()
    optimizer.step()


def play_one_game(max_steps=10000):
    s = env.reset()
    for _ in range(max_steps):
        pi, v = actor_critic(torch.as_tensor(s, dtype=torch.float32))
        a = pi.sample()

        new_s, r, done, info = env.step(a.item())

        memory.add(reward=r, log_prob=pi.log_prob(a), state_value=v)

        if done:
            break

        s = new_s


def train(num_of_games=1000):
    reward_hist = []

    for i in range(num_of_games):
        play_one_game()
        update()

        total_reward = sum(memory.rewards)
        ep_length = len(memory.rewards)
        reward_hist.append(total_reward)

        memory.clear()

        if i % 10 == 0:
            print(f"Episode {i}, length: {ep_length}, reward: {total_reward}")

    return reward_hist


def main():
    reward_hist = train(num_of_games=1500)
    env.close()

    plt.plot(reward_hist)
    plt.show()


if __name__ == "__main__":
    main()
