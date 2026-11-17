import os
import itertools
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import env_
from pathlib import Path
from models.ppo_agent import FSO_Agent

script_dir = Path(__file__).resolve().parent    # Get path of main.py's parent folder
os.chdir(script_dir)                            # Make parent folder the cwd

RUNS_DIR = "runs"

os.makedirs(RUNS_DIR, exist_ok=True)            # Make runs folder in cwd if no folder exists already

def save_ppo_graph(rewards_per_episode, PPO_HP):
    fig = plt.figure(2)
    plt.clf()
    mean_rewards = np.zeros(len(rewards_per_episode))
    for x in range(len(mean_rewards)):
        mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        
    plt.xlabel('Episodes')
    plt.ylabel('Mean HAPS Rewards')
    plt.plot(mean_rewards, color='green')
    
    fig.savefig(os.path.join(RUNS_DIR, f'{PPO_HP}.png'))
    plt.close(fig)

def save_ppo_reward_histogram(rewards_per_episode, PPO_HP, num_bins=30):
    fig = plt.figure(1, figsize=(8, 5))
    rewards = np.array(rewards_per_episode)

    # Plot histogram with outline and fill
    plt.hist(
        rewards,
        bins=num_bins,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
        label="Run Count",
    )

    # Summary statistics lines
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)

    plt.axvline(
        mean_reward,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_reward:.1f}",
    )
    plt.axvline(
        median_reward,
        color="green",
        linestyle="-.",
        linewidth=1.5,
        label=f"Median: {median_reward:.1f}",
    )

    # Labels and formatting
    plt.title(f"Episode Reward Distribution ({len(rewards)} Episodes)")
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")

    plt.tight_layout()

    # Save to directory and clear memory
    file_path = os.path.join(RUNS_DIR, f"{PPO_HP}_histogram.png")
    fig.savefig(file_path)
    plt.close(fig)

#=== Configuration Parameters ===#

CONTROLLER = "PPO"
PPO_TRAINING = True    # Toggle updating weights or not (Only used when CONTROLLER == "PPO")
PPO_CONTINUING = True  # Continue training using the .pt file in folder
RENDER = True         # Show the HAPS position and wind during each episode
MAX_EPISODES = None    # Set to "None" for indefinite runs

def train():
    controller = CONTROLLER.upper()
    env = gym.make("HAPS_FSO_Env-v0")
    agent_fso = None
    PPO_HP = "PPO"
    run_name = "PPO_station_keeping_v2"
    log_file = os.path.join(RUNS_DIR, f"{run_name}.log")
    actor_file = os.path.join(RUNS_DIR, f"{run_name}_actor.pt")
    critic_file = os.path.join(RUNS_DIR, f"{run_name}_critic.pt")
    rewards_history = []
    best_reward = float("-inf")

    try:
        if controller == "PPO":
            agent_fso = FSO_Agent(
                hyperparameter_set=PPO_HP,
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
            )
            if PPO_CONTINUING:
                if os.path.exists(actor_file) and os.path.exists(critic_file):
                    agent_fso.actor.load_state_dict(torch.load(actor_file, map_location="cpu"))
                    agent_fso.critic.load_state_dict(torch.load(critic_file, map_location="cpu"))
                elif not PPO_TRAINING:
                    raise FileNotFoundError("PPO checkpoint missing; cannot evaluate a trained PPO policy")
            elif not PPO_TRAINING:
                raise ValueError("PPO evaluation requires PPO_CONTINUING=True")
            if not PPO_TRAINING:
                agent_fso.actor.eval()

        episodes = itertools.count() if MAX_EPISODES is None else range(MAX_EPISODES)
        for episode in episodes:
            state, info = env.reset()
            terminated = truncated = False
            episode_reward = 0.0
            position_errors = []
            initial_error = info["position_error"]
            final_error = initial_error

            while not (terminated or truncated):    
                action, log_prob, value = agent_fso.select_action(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                if RENDER:
                    env.render()

                episode_reward += float(reward)
                position_error = info["position_error"]
                position_errors.append(position_error)
                final_error = position_error

                if controller == "PPO" and PPO_TRAINING:
                    agent_fso.memory.store_memory(
                        state, action, log_prob, value, reward, terminated or truncated
                    )
                state = next_state

            if controller == "PPO" and PPO_TRAINING:
                agent_fso.optimize()

            rewards_history.append(episode_reward)
            mean_position_error = (
                float(np.mean(position_errors)) if position_errors else final_error
            )
            label = "New Best Reward" if episode_reward > best_reward else "Reward"
            log_message = (
                f"{controller} Ep {episode} - {label}: {episode_reward:.1f}"
            )
            print(log_message)
            with open(log_file, "a", encoding="utf-8") as file:
                file.write(log_message + "\n")

            if episode_reward > best_reward:
                best_reward = episode_reward
                if controller == "PPO" and PPO_TRAINING:
                    torch.save(agent_fso.actor.state_dict(), actor_file)
                    torch.save(agent_fso.critic.state_dict(), critic_file)

            if episode % 5 == 0:
                save_ppo_graph(rewards_history, run_name)
                save_ppo_reward_histogram(rewards_history, run_name)
    finally:
        env.close()


if __name__ == "__main__":
    train()
