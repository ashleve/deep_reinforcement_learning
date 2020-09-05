import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler
from torch.optim import Adam
from torch.distributions import Categorical, Normal
import wandb

wandb.init(project="PPO_continuous", group="LunarLanderContinuous-v2", job_type="eval")

config = wandb.config
# config.env_name = 'LunarLander-v2'
# config.env_name = 'LunarLanderContinuous-v2'
config.env_name = 'BipedalWalker-v3'
# config.env_name = 'HalfCheetah-v2'
# config.env_name = 'CartPole-v0'

config.gamma = 0.99
# config.lamb = 0.95
config.batch_size = 512
config.memory_size = 8192
config.hidden_size = 64
config.actor_lr = 0.01
config.critic_lr = 0.01
config.ppo_multiple_epochs = 3
config.eps = 0.2
config.grad_clip_norm = 0.5

env = gym.make(config.env_name)

config.obs_space = env.observation_space.shape[0]
config.action_space = env.action_space.shape[0]

print(env.action_space.high)
print(env.action_space.low)
print(config)


config.seed = 123
torch.manual_seed(config.seed)


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
        self.linear2 = nn.Linear(config.hidden_size, 12)

        self.log_std = nn.Linear(12, config.action_space)
        self.mean = nn.Linear(12, config.action_space)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5

    def forward(self, state):
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)

        # mean = torch.tanh(self.mean(x))
        mean = self.mean(x)

        log_std = self.log_std(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp

        std = log_std.exp()

        pi = Normal(mean, std)

        return pi


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(config.obs_space, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, 12)

        self.value_layer = nn.Linear(12, 1)

    def forward(self, state):
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        v = self.value_layer(x)

        return v


actor = Actor()
critic = Critic()
optimizer_actor = Adam(actor.parameters(), lr=config.actor_lr)
optimizer_critic = Adam(critic.parameters(), lr=config.critic_lr)
memory = Memory()


def compute_GAE(rewards, state_values, gamma, lamb):
    """
        Computes Generalized Advantage Estimations for a single trajectory.
        Params:
            gamma - Discount.
    """
    gae = []
    running_sum = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * state_values[i + 1] - state_values[i]
        running_sum = delta + gamma * lamb * running_sum
        gae.insert(0, running_sum)

    return gae


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
    # compute critic loss
    v = critic(states)
    critic_loss = F.mse_loss(rewards_to_go, v)
    # critic_loss = F.smooth_l1_loss(rewards, v)

    # compute actor loss
    pi = actor(states)
    log_probs = pi.log_prob(actions)
    ratio = torch.exp(log_probs - old_log_probs)  # exp(log_prob - old_log_prob) = (prob / old_prob)
    clip = torch.clamp(ratio, 1 - config.eps, 1 + config.eps)
    actor_loss = -torch.min(ratio * adv, clip * adv).mean()

    return actor_loss, critic_loss, ratio.mean()


def update():
    states = torch.as_tensor(memory.states, dtype=torch.float32)
    actions = torch.as_tensor(memory.actions, dtype=torch.float32)
    state_values = critic(states)

    # compute rewards-to-go
    rewards_to_go = compute_r2g(memory.rewards, memory.done, gamma=config.gamma)
    rewards_to_go = torch.as_tensor(rewards_to_go, dtype=torch.float32)
    rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-5)  # normalize
    rewards_to_go = rewards_to_go.unsqueeze(1)  # transform to shape [1, memory_size]

    # compute advantage estimations
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
        batch_sampler = BatchSampler(sampler, batch_size=config.batch_size, drop_last=True)

        # execute epoch
        for indices in batch_sampler:
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_rewards_to_go = rewards_to_go[indices]
            batch_adv = adv[indices]
            batch_old_log_probs = old_log_probs[indices]

            actor_loss, critic_loss, _ = compute_loss(
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
    actor_loss, critic_loss, ratio = compute_loss(states, actions, rewards_to_go, adv, old_log_probs)
    wandb.log({
        "actor loss": actor_loss,
        "critic loss": critic_loss,
        "ppo prob ratio": ratio
    })


# gae = compute_GAE(rewards, state_values, gamma=config.gamma, lamb=config.lamb)
# memory.adv_estimations.extend(gae)


def train(num_of_games=1000, max_steps=10000):

    total_number_of_steps = 0

    for i in range(1, num_of_games):
        s = env.reset()

        total_ep_reward = 0
        for j in range(max_steps):
            pi = actor(torch.as_tensor(s, dtype=torch.float32))
            a = torch.clamp(pi.sample(), -1, 1).numpy()

            new_s, r, done, info = env.step(a)

            done = 0 if done else 1
            memory.add(state=s, action=a, reward=r, done=done)

            total_ep_reward += r
            total_number_of_steps += 1

            if memory.size >= config.memory_size:
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
        })


def main():
    wandb.watch(actor)
    wandb.watch(critic)

    train(num_of_games=1500)

    # torch.save(actor_critic.state_dict(), "model.h5")
    # wandb.save('model.h5')


if __name__ == "__main__":
    main()
    env.close()
