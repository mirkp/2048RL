import gymnasium as gym
import gymnasium_2048  # registers the env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

ENV_NAME = "gymnasium_2048/TwentyFortyEight-v0"
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
REPLAY_SIZE = 50_000
MIN_REPLAY_SIZE = 2_000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 50_000
TARGET_UPDATE_FREQ = 1_000
NUM_EPISODES = 1_000
MAX_STEPS_PER_EP = 10_000
MODEL_PATH = "dqn_2048.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim=16, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.from_numpy(np.stack(states)).float().to(DEVICE),
            torch.tensor(actions).long().unsqueeze(1).to(DEVICE),
            torch.tensor(rewards).float().unsqueeze(1).to(DEVICE),
            torch.from_numpy(np.stack(next_states)).float().to(DEVICE),
            torch.tensor(dones).float().unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)

def preprocess(obs):
    """
    Converts gymnasium_2048 observations to a (16,) float vector
    Handles:
      - dict observations
      - (4,4)
      - (4,4,1)
      - (4,4,16) one-hot encoding
      - (16,)
    """

    # Unwrap dict
    if isinstance(obs, dict):
        obs = obs.get("board", obs)

    board = np.asarray(obs, dtype=np.float32)

    # Case 1: one-hot encoded (4,4,16)
    if board.shape == (4, 4, 16):
        # Decode: index of hot channel = log2(tile)
        board = board.argmax(axis=-1)

    # Case 2: (4,4,1)
    elif board.shape == (4, 4, 1):
        board = board.squeeze(-1)

    # Case 3: already flat
    if board.shape == (16,):
        flat = board

    # Case 4: normal board
    elif board.shape == (4, 4):
        flat = board.reshape(-1)

    else:
        raise ValueError(f"Unexpected board shape: {board.shape}")

    return flat.astype(np.float32)

def select_action(state, policy_net, epsilon, action_space):
    """
    Epsilon-greedy action selection
    """
    if random.random() < epsilon:
        return action_space.sample()

    with torch.no_grad():
        state_tensor = (
            torch.from_numpy(state)
            .float()
            .unsqueeze(0)
            .to(DEVICE)
        )
        q_values = policy_net(state_tensor)
        return q_values.argmax(dim=1).item()
    
def cluster_reward(board):
    """
    Gives a small bonus if high tiles (>=8) are adjacent to each other.
    Encourages the agent to cluster high tiles.
    """
    board = board.reshape(4, 4)
    reward = 0
    for i in range(4):
        for j in range(4):
            if board[i, j] >= 8:  # consider "high" tiles
                # check right and down neighbors to avoid double-counting
                for di, dj in [(0, 1), (1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 4 and 0 <= nj < 4:
                        reward += (board[i, j] + board[ni, nj]) / 2
    return reward

def train():
    env = gym.make(ENV_NAME)

    policy_net = DQN().to(DEVICE)
    target_net = DQN().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_SIZE)

    epsilon = EPS_START
    step_count = 0

    for episode in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        state = preprocess(obs)

        total_reward = 0
        done = False
        steps = 0
        max_tile_achieved = int(np.max(state))

        while not done and steps < MAX_STEPS_PER_EP:
            steps += 1
            step_count += 1

            # Epsilon decay
            epsilon = max(
                EPS_END,
                EPS_START - step_count / EPS_DECAY_STEPS
            )

            action = select_action(state, policy_net, epsilon, env.action_space)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = preprocess(next_obs)

            # Add clustering/top-tile bonus
            cluster_bonus = cluster_reward(next_state)
            reward += 0.01 * cluster_bonus  # small scale so agent still mostly follows env reward

            replay.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            max_tile_achieved = max(max_tile_achieved, int(np.max(next_state)))

            # Train
            if len(replay) >= MIN_REPLAY_SIZE:
                states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)

                q_vals = policy_net(states).gather(1, actions)

                with torch.no_grad():
                    next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
                    target = rewards + GAMMA * next_q * (1 - dones)

                loss = nn.MSELoss()(q_vals, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if step_count % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(
            f"Episode {episode:4d} | "
            f"Reward: {total_reward:8.1f} | "
            f"Max Tile: {max_tile_achieved:4d} | "
            f"Epsilon: {epsilon:.3f}"
        )

        # Save periodically
        if episode % 50 == 0:
            torch.save(policy_net.state_dict(), MODEL_PATH)

    torch.save(policy_net.state_dict(), MODEL_PATH)
    env.close()
    print(f"\nTraining complete. Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
