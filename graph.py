import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure folder exists for saving graphs
os.makedirs("graphs", exist_ok=True)


def plot_rewards(rewards, filename="graphs/rewards.png"):
    """Plots episode rewards over time"""
    plt.figure(figsize=(10,5))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_max_tiles(max_tiles, filename="graphs/max_tiles.png"):
    """Plots maximum tile achieved per episode"""
    plt.figure(figsize=(10,5))
    plt.plot(max_tiles, label="Max Tile")
    plt.xlabel("Episode")
    plt.ylabel("Max Tile")
    plt.title("Maximum Tile Per Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_moving_average(values, window=20, filename="graphs/moving_average.png"):
    """Plots moving average of a list of values"""
    moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10,5))
    plt.plot(range(window-1, len(values)), moving_avg, label=f"Moving Avg (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title(f"Moving Average over {window} episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_value_distribution(values_list, filename="graphs/value_distribution.png"):
    """Plots histogram/distribution of values collected (for example, returns or advantages)"""
    plt.figure(figsize=(10,5))
    plt.hist(values_list, bins=50, alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Value Distribution")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_action_probabilities(action_probs_list, filename="graphs/action_probabilities.png"):
    """Plots average probability of each action over time"""
    action_probs = np.array(action_probs_list)
    plt.figure(figsize=(10,5))
    for i in range(action_probs.shape[1]):
        plt.plot(action_probs[:,i], label=f"Action {i}")
    plt.xlabel("Step")
    plt.ylabel("Action Probability")
    plt.title("Action Probabilities Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
