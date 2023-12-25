import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 50)
        self.fc2 = nn.Linear(50, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.99, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state)
                q_values = self.model(state)
                return int(torch.argmax(q_values).item())

    def train(self, state, action, next_state, reward, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor([action])
        reward = torch.FloatTensor([reward])
        target = reward + self.gamma * torch.max(self.model(next_state))
        target = target.unsqueeze(0) if not done else reward.unsqueeze(0)

        self.optimizer.zero_grad()
        q_values = self.model(state)
        loss = self.criterion(q_values.gather(1, action.unsqueeze(1)), target.unsqueeze(1))
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Example Usage:
state_size = 64  # 8x8 grid
action_size = 4  # possible actions (left, right, up, down)
agent = QLearningAgent(state_size, action_size)

# Training loop
env_name = 'FrozenLake8x8-v1'
env = gym.make(env_name)

for episode in range(1000):
    state = env.reset()  # Assuming 'env' is your environment
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.train(state, action, next_state, reward, done)

        total_reward += reward
        state = next_state

        if done:
            break

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
