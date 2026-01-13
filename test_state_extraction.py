import gymnasium as gym
import gymnasium_2048
import numpy as np
import time
import pygame

def obs_to_board(obs):
    obs = np.array(obs)
    board = np.zeros((4,4), dtype=int)
    for i in range(4):
        for j in range(4):
            idx = int(np.argmax(obs[i, j]))
            if idx > 0:
                board[i, j] = 2 ** idx
    return board

def print_board(board):
    print("-" * 25)
    for row in board:
        print(" ".join(f"{v:5d}" if v != 0 else "    ." for v in row))
    print("-" * 25)

def debug_game():
    env = gym.make(
        "gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
        render_mode="human",
        size=4,
        max_pow=16
    )

    obs, _ = env.reset()
    env.render()  # force initial render
    time.sleep(0.2)

    paused = False
    step = 0

    print("\nControls: P = pause/resume | ESC = quit\n")

    while True:
        pygame.event.pump()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_p]:
            paused = not paused
            print("\nPAUSED" if paused else "\nRESUMED")
            time.sleep(0.3)

        if keys[pygame.K_ESCAPE]:
            break

        if paused:
            time.sleep(0.05)
            continue

        # ---- STEP ENV ----
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        # ---- FORCE RENDER ----
        env.render()
        time.sleep(0.15)  # ensures render catches up

        # ---- PRINT EXACT STATE ----
        board = obs_to_board(obs)
        print(f"\nStep {step}")
        print_board(board)
        print(f"Action: {action}, Reward: {reward}")

        step += 1

        if terminated or truncated:
            print("\nGAME OVER")
            print_board(obs_to_board(obs))
            time.sleep(2)
            break

    env.close()

if __name__ == "__main__":
    debug_game()
