import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


class DQN:

    def __init__(self):
        self.env = gym.make('CartPole-v1')

        self.observation_space = self.env.observation_space.shape
        self.action_space = self.env.action_space.n

        self.LEARNING_RATE = 0.01
        self.MEMORY_SIZE = 8192
        self.BATCH_SIZE = 256
        self.EPISODES = 1000
        self.MAX_STEPS = 500
        self.EPSILON = 0.1
        self.GAMMA = 1.0

        self.memory = deque(maxlen=self.MEMORY_SIZE)

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(12, input_shape=self.observation_space, activation="relu"))
        model.add(Dense(12, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))
        model.compile(loss="mse", optimizer=RMSprop(lr=self.LEARNING_RATE))
        return model

    def remember(self, experience):
        self.memory.append(experience)

    def reshape_state(self, s):
        return np.array(s).reshape((1, 4))

    def choose_max_action(self, s):
        q_values = self.model.predict(self.reshape_state(s))[0]
        return np.argmax(q_values)

    def choose_action(self, s):
        if np.random.random() < self.EPSILON:
            return self.env.action_space.sample()
        else:
            return self.choose_max_action(s)

    def get_random_minibatch(self):
        idx = np.random.choice(len(self.memory), size=self.BATCH_SIZE, replace=False)
        return np.array(self.memory)[idx]

    def experience_replay(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        minibatch = self.get_random_minibatch()

        states = np.stack(minibatch[:, 0], axis=0)
        actions = minibatch[:, 1]
        rewards = minibatch[:, 2]
        next_states = np.stack(minibatch[:, 3], axis=0)
        done = minibatch[:, 4]

        state_values = self.model.predict(states)
        next_state_values = self.model.predict(next_states)

        estimated_max_q_values = rewards + self.GAMMA * np.max(next_state_values, axis=1)

        for i in range(len(state_values)):
            # Terminal states have no future, so we should set their Q-value to their immediate reward.
            state_values[i][actions[i]] = estimated_max_q_values[i] if done[i] is False else rewards[i]

        self.model.fit(states, state_values, verbose=0, epochs=1)

    def train(self):
        reward_hist = []

        for e in range(self.EPISODES):

            s = self.env.reset()
            total_reward = 0

            for i in range(self.MAX_STEPS):

                a = self.choose_action(s)
                s_next, reward, done, _ = self.env.step(a)

                experience = (s, a, reward, s_next, done)
                self.remember(experience)

                if done:
                    reward_hist.append(total_reward)
                    break

                s = s_next
                total_reward += reward

            self.experience_replay()
            print(f"Episode: {e}, length: {i} reward: {total_reward}")

        return reward_hist


def main():
    dqn = DQN()
    reward_hist = dqn.train()

    plt.plot(reward_hist)
    plt.show()


if __name__ == "__main__":
    main()
