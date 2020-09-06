import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler, BatchSampler
from torch.optim import Adam
from torch.distributions import Normal
import wandb
# import numpy as np
# import pybullet_envs
import time


wandb.init(project="PPO_continuous2", group="lunar_batch_256_mem_4096_epochs_10_grad_clip", job_type="eval")

config = wandb.config
# config.env_name = 'LunarLander-v2'
config.env_name = 'LunarLanderContinuous-v2'
# config.env_name = 'BipedalWalker-v3'
# config.env_name = 'MountainCarContinuous-v0'
# config.env_name = 'MinitaurBulletEnv-v0'
# config.env_name = 'HumanoidBulletEnv-v0'
# config.env_name = 'AntBulletEnv-v0'
# config.env_name = 'HalfCheetahBulletEnv-v0'
# config.env_name = 'CartPoleContinuousBulletEnv-v0'
# config.env_name = 'Walker2DBulletEnv-v0'
# config.env_name = 'HopperBulletEnv-v0'
# config.env_name = 'Pendulum-v0'
# config.env_name = 'HalfCheetah-v2'
# config.env_name = 'CartPole-v0'

env = gym.make(config.env_name)

config.gamma = 0.99
config.batch_size = 256
config.memory_size = 4096
config.hidden_size = 64
config.actor_lr = 0.0003
config.critic_lr = 0.0003
config.ppo_multiple_epochs = 10
config.eps = 0.2
config.grad_clip_norm = 0.5
config.entropy_weight = 0.01
config.max_timesteps = 1500

config.obs_space = env.observation_space.shape[0]
config.action_space = env.action_space.shape[0]
config.action_high = env.action_space.high[0]
config.action_low = env.action_space.low[0]

print(config.obs_space)
print(config.action_space)
print(env.action_space.high)
print(env.action_space.low)


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.log_probs = []
        self.size = 0

    def add(self, state, action, reward, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.done.append(done)
        self.log_probs.append(log_prob)
        self.size += 1

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.done.clear()
        self.log_probs.clear()
        self.size = 0


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(config.obs_space, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, config.action_space),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.zeros(config.action_space), requires_grad=True)

    def forward(self, state):
        mean = self.actor(state)
        std = self.log_std.expand_as(mean).exp()
        pi = Normal(mean, std)
        a = pi.sample()
        return a, pi.log_prob(a).sum(1), pi


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(config.obs_space, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        v = self.critic(state)
        return torch.squeeze(v)


actor = Actor()
critic = Critic()
optimizer_actor = Adam(actor.parameters(), lr=config.actor_lr)
optimizer_critic = Adam(critic.parameters(), lr=config.critic_lr)
memory = Memory()


def compute_r2g(rewards, done, gamma):
    """
        Computes discounted rewards-to-go.
        Params:
            gamma - Discount.
    """
    rewards2go = []
    running_sum = 0
    for r, is_terminal in zip(reversed(rewards), reversed(done)):
        running_sum = r + gamma * running_sum * is_terminal
        rewards2go.insert(0, running_sum)

    return rewards2go


def compute_loss(states, actions, rewards_to_go, adv, old_log_probs):

    # compute critic loss
    v = critic(states)
    critic_loss = (rewards_to_go - v).pow(2)

    # compute actor loss
    _, _, pi = actor(states)
    log_probs = pi.log_prob(actions).sum(1)
    ratio = torch.exp(log_probs - old_log_probs)  # exp(log_prob - old_log_prob) = (prob / old_prob)
    clip = torch.clamp(ratio, 1 - config.eps, 1 + config.eps)
    actor_loss = -torch.min(ratio * adv, clip * adv)

    # compute entropy
    entropy = pi.entropy().sum(1)
    actor_loss -= config.entropy_weight * entropy

    return actor_loss.mean(), critic_loss.mean(), entropy.mean(), ratio.mean()


def update():
    start = time.time()

    # compute rewards-to-go
    rewards_to_go = compute_r2g(memory.rewards, memory.done, gamma=config.gamma)

    # prepare data
    states = torch.squeeze(torch.stack(memory.states), 1).detach()
    actions = torch.squeeze(torch.stack(memory.actions), 1).detach()
    old_log_probs = torch.squeeze(torch.stack(memory.log_probs), 1).detach()
    rewards_to_go = torch.tensor(rewards_to_go).detach()
    state_values = critic(states).detach()

    # normalize rewards-to-go
    rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-5)

    # compute advantage estimations
    adv = (rewards_to_go - state_values)

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

            actor_loss, critic_loss, _, _ = compute_loss(
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
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), config.grad_clip_norm)
            optimizer_actor.step()

    # log stats
    actor_loss, critic_loss, entropy, ratio = compute_loss(states, actions, rewards_to_go, adv, old_log_probs)
    end = time.time()
    wandb.log({
        "actor loss": actor_loss,
        "critic loss": critic_loss,
        "ppo prob ratio": ratio,
        "entropy": entropy,
        "loss computation time": end - start
    })


def train(num_of_games=10000, max_steps=10000):
    total_number_of_steps = 0

    for i in range(1, num_of_games):
        s = env.reset()

        total_ep_reward = 0
        for j in range(max_steps):

            s = torch.FloatTensor(s.reshape(1, -1))
            # s = torch.FloatTensor(s)

            a, log_prob, pi = actor(s)

            new_s, r, done, info = env.step(a.numpy().flatten())

            done = 0 if done else 1
            memory.add(state=s, action=a, reward=r, done=done, log_prob=log_prob)

            total_ep_reward += r
            total_number_of_steps += 1

            if memory.size >= config.memory_size:
                memory.rewards[-1] += config.gamma * critic(torch.FloatTensor(new_s)).item()  # bootstrap future reward
                update()
                memory.clear()

            if done == 0:
                break

            s = new_s

        print(f"Episode {i}, length: {j}")

        wandb.log({
            "total number of steps": total_number_of_steps,
            "total number of games": i,
            "total reward": total_ep_reward,
            "episode length": j
        }, step=total_number_of_steps)


def main():
    # wandb.watch(actor)
    # wandb.watch(critic)

    train(num_of_games=1000000, max_steps=config.max_timesteps)

    # torch.save(actor_critic.state_dict(), "model.h5")
    # wandb.save('model.h5')


if __name__ == "__main__":
    main()
    env.close()
