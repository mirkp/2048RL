import gymnasium as gym
import gymnasium_2048
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from graph import plot_rewards, plot_max_tiles, plot_moving_average, plot_value_distribution

ENV_NAME = "gymnasium_2048/TwentyFortyEight-v0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
LR = 3e-4
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
NUM_EPISODES = 1000
MAX_STEPS_PER_EP = 10000
UPDATE_EVERY = 2048  # timesteps before PPO update

# -----------------------------
# Preprocess board
# -----------------------------
def preprocess(obs):
    if isinstance(obs, dict):
        obs = obs.get("board", obs)

    board = np.asarray(obs, dtype=np.float32)

    # One-hot (4,4,16)
    if board.shape == (4, 4, 16):
        return board.astype(np.float32)
    elif board.shape == (4, 4):
        # Convert to log2 encoding, then one-hot along channel
        log_board = np.log2(np.clip(board, 1, None))
        one_hot = np.zeros((4, 4, 16), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                idx = int(log_board[i, j])
                if idx < 16:
                    one_hot[i, j, idx] = 1.0
        return one_hot
    elif board.shape == (4, 4, 1):
        board = board.squeeze(-1)
        return preprocess(board)
    else:
        raise ValueError(f"Unexpected board shape: {board.shape}")

# -----------------------------
# CNN Actor-Critic
# -----------------------------
class CNNActorCritic(nn.Module):
    def __init__(self, action_dim=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2),  # 4x4x16 -> 3x3x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),  # 3x3x32 -> 2x2x64
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*2*64, 256),
            nn.ReLU()
        )
        self.policy = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return self.policy(x), self.value(x)

# -----------------------------
# PPO Agent
# -----------------------------
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.net = CNNActorCritic().to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)

    def select_action(self, state):
        state_tensor = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
        logits, value = self.net(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.detach(), value.detach()

    def compute_gae(self, rewards, values, dones, next_value, gamma=GAMMA, lam=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, memory):
        states = torch.stack(memory['states']).to(DEVICE).permute(0,3,1,2)
        actions = torch.tensor(memory['actions']).to(DEVICE)
        old_log_probs = torch.stack(memory['log_probs']).to(DEVICE)
        returns = torch.tensor(memory['returns'], dtype=torch.float32).to(DEVICE)
        values = torch.stack(memory['values']).squeeze(1).to(DEVICE)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):  # PPO epochs
            logits, value = self.net(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = VALUE_COEF * (returns - value.squeeze(1)).pow(2).mean()
            loss = policy_loss + value_loss - ENTROPY_COEF * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# -----------------------------
# Training Loop
# -----------------------------
def train():
    env = gym.make(ENV_NAME)
    agent = PPOAgent(env)

    episode_rewards = []
    episode_max_tiles = []

    for episode in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        state = preprocess(obs)

        memory = {'states': [], 'actions': [], 'log_probs': [], 'values': [], 'rewards': [], 'dones': []}
        total_reward = 0
        done = False
        steps = 0

        # Correct max tile computation
        max_tile_achieved = 0

        while not done and steps < MAX_STEPS_PER_EP:
            steps += 1
            action, log_prob, value = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess(next_obs)

            # Optional: clustering reward bonus
            cluster_bonus = np.sum(next_state[:,:,8:] > 0)  # reward high tiles
            reward += 0.01 * cluster_bonus

            # Store memory
            memory['states'].append(torch.from_numpy(state).float())
            memory['actions'].append(action)
            memory['log_probs'].append(log_prob)
            memory['values'].append(value)
            memory['rewards'].append(reward)
            memory['dones'].append(done)

            # Correct max tile: ignore empty channel index 0
            tile_indices = next_state.argmax(axis=-1)
            tile_indices_nonzero = tile_indices[tile_indices > 0]
            if len(tile_indices_nonzero) > 0:
                max_tile_achieved = max(max_tile_achieved, 2 ** int(tile_indices_nonzero.max()))

            state = next_state
            total_reward += reward

            # PPO update
            if len(memory['states']) >= UPDATE_EVERY or done:
                with torch.no_grad():
                    next_state_tensor = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
                    _, next_value = agent.net(next_state_tensor)
                returns = agent.compute_gae(memory['rewards'], memory['values'], memory['dones'], next_value)
                memory['returns'] = returns
                agent.update(memory)
                memory = {'states': [], 'actions': [], 'log_probs': [], 'values': [], 'rewards': [], 'dones': []}

        print(f"Episode {episode:4d} | Reward: {total_reward:8.1f} | Max Tile: {max_tile_achieved:4d}")

        episode_rewards.append(total_reward)
        episode_max_tiles.append(max_tile_achieved)

    # -----------------------------
    # Save graphs after training
    # -----------------------------
    plot_rewards(episode_rewards, filename="graphs/rewards.png")
    plot_max_tiles(episode_max_tiles, filename="graphs/max_tiles.png")
    plot_moving_average(episode_rewards, window=20, filename="graphs/rewards_moving_avg.png")
    plot_moving_average(episode_max_tiles, window=20, filename="graphs/max_tiles_moving_avg.png")

    print("Training complete. Graphs saved in 'graphs/' folder.")
    env.close()

if __name__ == "__main__":
    train()
