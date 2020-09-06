from collections import deque

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler
from torch.optim import Adam
from torch.distributions import Categorical, Normal
import wandb
import numpy as np
import pybullet_envs


wandb.init(project="PPO_continuous2", group="mem_12000", job_type="eval")

config = wandb.config
# config.env_name = 'LunarLander-v2'
# config.env_name = 'LunarLanderContinuous-v2'
# config.env_name = 'BipedalWalker-v3'
# config.env_name = 'MountainCarContinuous-v0'
# config.env_name = 'MinitaurBulletEnv-v0'
# config.env_name = 'HumanoidBulletEnv-v0'
# config.env_name = 'AntBulletEnv-v0'
# config.env_name = 'HalfCheetahBulletEnv-v0'
# config.env_name = 'CartPoleContinuousBulletEnv-v0'
config.env_name = 'Walker2DBulletEnv-v0'
# config.env_name = 'HopperBulletEnv-v0'
# config.env_name = 'Pendulum-v0'
# config.env_name = 'HalfCheetah-v2'
# config.env_name = 'CartPole-v0'


# env = e.MinitaurBulletEnv(render=False)


config.gamma = 0.99
# config.lamb = 0.95
config.batch_size = 32
config.memory_size = 2048
config.hidden_size = 64
config.actor_lr = 0.0003
config.critic_lr = 0.0003
config.ppo_multiple_epochs = 10
config.eps = 0.2
config.grad_clip_norm = 0.5
config.entropy_weight = 0.1

env = gym.make(config.env_name)

config.obs_space = env.observation_space.shape[0]
config.action_space = env.action_space.shape[0]
config.action_high = env.action_space.high[0]
config.action_low = env.action_space.low[0]

print(config.action_space)
print(env.action_space.high)
print(env.action_space.low)
# print(config)


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    # def action(self, action: np.ndarray) -> np.ndarray:
    #     """Change the range (-1, 1) to (low, high)."""
    #     low = self.action_space.low
    #     high = self.action_space.high
    #
    #     # action = np.nan_to_num(action)
    #
    #     scale_factor = (high - low) / 2
    #     reloc_factor = high - scale_factor
    #
    #     action = action * scale_factor + reloc_factor
    #     action = np.clip(action, low, high)
    #
    #     # print(action)
    #
    #     return action

    def action(self, action):
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    # def reverse_action(self, action: np.ndarray) -> np.ndarray:
    #     """Change the range (low, high) to (-1, 1)."""
    #     low = self.action_space.low
    #     high = self.action_space.high
    #
    #     scale_factor = (high - low) / 2
    #     reloc_factor = high - scale_factor
    #
    #     action = (action - reloc_factor) / scale_factor
    #     action = np.clip(action, -1.0, 1.0)
    #
    #     return action


env = ActionNormalizer(env)

# config.seed = 123
# torch.manual_seed(config.seed)


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


rms = RunningMeanStd()


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.size = 0

    def add(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.done.append(done)
        self.size += 1

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.done.clear()
        self.size = 0


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(config.obs_space, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.log_std = nn.Parameter(torch.zeros(config.action_space), requires_grad=True)

        # self.log_std = nn.Linear(config.hidden_size, config.action_space)
        self.mean = nn.Linear(config.hidden_size, config.action_space)

        # self.reset_parameters()

        # self.LOG_STD_MAX = 0
        # self.LOG_STD_MIN = -3

    def forward(self, state):
        x = self.linear1(state)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.tanh(x)

        mean = torch.tanh(self.mean(x))

        # log_std = torch.tanh(self.log_std)
        # log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp
        # print(self.log_std.shape)

        std = self.log_std.expand_as(mean).exp()

        pi = Normal(mean, std)

        return pi

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.mean.weight)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(config.obs_space, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.value_layer = nn.Linear(config.hidden_size, 1)

        # self.reset_parameters()

    def forward(self, state):
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        v = self.value_layer(x)

        return v

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.value_layer.weight)


actor = Actor()
critic = Critic()
optimizer_actor = Adam(actor.parameters(), lr=config.actor_lr)
optimizer_critic = Adam(critic.parameters(), lr=config.critic_lr)
memory = Memory()


# def compute_GAE(rewards, state_values, gamma, lamb):
#     """
#         Computes Generalized Advantage Estimations for a single trajectory.
#         Params:
#             gamma - Discount.
#     """
#     gae = []
#     running_sum = 0
#     for i in reversed(range(len(rewards))):
#         delta = rewards[i] + gamma * state_values[i + 1] - state_values[i]
#         running_sum = delta + gamma * lamb * running_sum
#         gae.insert(0, running_sum)
#
#     return gae

def compute_gae(
    rewards: list,
    masks: list,
    values: list,
    gamma: float,
    tau: float
):
    """Compute gae."""
    next_value = 0
    values = values + [next_value]
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        delta = (
            rewards[step]
            + gamma * values[step + 1] * masks[step]
            - values[step]
        )
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae)

    return list(returns)


def compute_r2g(rewards, done, gamma):
    """
        Computes discounted rewards-to-go.
        Params:
            gamma - Discount.
    """
    rewards2go = []
    running_sum = 0
    for r, is_terminal in zip(rewards[::-1], done[::-1]):
        running_sum = r + gamma * running_sum * is_terminal
        rewards2go.insert(0, running_sum)

    return rewards


def compute_loss(states, actions, rewards_to_go, adv, old_log_probs):

    # normalize advantage estimations
    adv = (adv - adv.mean()) / (adv.std() + 1e-7)

    # compute critic loss
    v = critic(states)
    critic_loss = F.mse_loss(rewards_to_go, v)

    # compute actor loss
    pi = actor(states)
    log_probs = pi.log_prob(actions)
    ratio = torch.exp(log_probs - old_log_probs)  # exp(log_prob - old_log_prob) = (prob / old_prob)
    clip = torch.clamp(ratio, 1 - config.eps, 1 + config.eps)
    actor_loss = -torch.min(ratio * adv, clip * adv).mean()

    # compute entropy
    entropy = pi.entropy().mean()

    return actor_loss, critic_loss, entropy, ratio.mean()


def update():
    states = torch.as_tensor(memory.states, dtype=torch.float32)
    actions = torch.as_tensor(memory.actions, dtype=torch.float32)
    state_values = critic(states)

    # compute rewards-to-go
    rewards_to_go = compute_r2g(memory.rewards, memory.done, gamma=config.gamma)
    rewards_to_go = torch.as_tensor(rewards_to_go, dtype=torch.float32)
    rewards_to_go = rewards_to_go.unsqueeze(1)  # transform to shape [1, memory_size]

    # compute gae
    # gae = compute_gae(memory.rewards, memory.done, state_values.reshape(-1).tolist(), gamma=config.gamma, tau=config.lamb)
    # gae = torch.as_tensor(gae, dtype=torch.float32)
    # gae = gae.unsqueeze(1)  # transform to shape [1, memory_size]

    # compute advantage estimations
    # adv = (gae - state_values)
    adv = (rewards_to_go - state_values)

    # compute old log probabilities
    pi = actor(states)
    old_log_probs = pi.log_prob(actions)

    # detach
    rewards_to_go = rewards_to_go.detach()
    adv = adv.detach()
    old_log_probs = old_log_probs.detach()

    # learn
    for _ in range(config.ppo_multiple_epochs):

        # create sampler
        sampler = SubsetRandomSampler(range(memory.size))
        batch_sampler = BatchSampler(sampler, batch_size=config.batch_size, drop_last=False)

        # execute epoch
        for indices in batch_sampler:
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_rewards_to_go = rewards_to_go[indices]
            batch_adv = adv[indices]
            batch_old_log_probs = old_log_probs[indices]

            actor_loss, critic_loss, entropy, _ = compute_loss(
                batch_states,
                batch_actions,
                batch_rewards_to_go,
                batch_adv,
                batch_old_log_probs
            )

            # update critic
            optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), config.grad_clip_norm)
            optimizer_critic.step()

            # update actor
            optimizer_actor.zero_grad()
            actor_loss -= entropy * config.entropy_weight
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), config.grad_clip_norm)
            optimizer_actor.step()

    # log stats
    actor_loss, critic_loss, entropy, ratio = compute_loss(states, actions, rewards_to_go, adv, old_log_probs)
    wandb.log({
        "actor loss": actor_loss,
        "critic loss": critic_loss,
        "ppo prob ratio": ratio,
        "entropy": entropy
    })


def train(num_of_games=10000, max_steps=10000):
    total_number_of_steps = 0

    for i in range(1, num_of_games):
        s = env.reset()

        total_ep_reward = 0
        for j in range(max_steps):

            rms.update(s)
            s = np.array(s)
            s = np.clip((s - rms.mean) / np.sqrt(rms.var + 1e-7), -10, 10)

            pi = actor(torch.as_tensor(s, dtype=torch.float32))
            a = pi.sample().detach().numpy()

            # print(s, a)
            # a = torch.clamp(pi.sample(), config.action_low, config.action_high).numpy()

            new_s, r, done, info = env.step(a)

            done = 0 if done else 1
            memory.add(state=s, action=a, reward=r, done=done)

            total_ep_reward += r
            total_number_of_steps += 1

            if memory.size >= config.memory_size:
                memory.rewards[-1] += critic(torch.as_tensor(new_s, dtype=torch.float32)).item()  # bootstrap future reward
                update()
                memory.clear()

            if done == 0:
                break

            s = new_s

        print(f"Episode {i}, length: {j}")

        wandb.log({
            "total number of steps": total_number_of_steps,
            "total reward": total_ep_reward,
            "episode length": j
        }, step=total_number_of_steps)


def main():
    # wandb.watch(actor)
    # wandb.watch(critic)

    train(num_of_games=150000, max_steps=50000)

    # torch.save(actor_critic.state_dict(), "model.h5")
    # wandb.save('model.h5')


if __name__ == "__main__":
    main()
    env.close()
