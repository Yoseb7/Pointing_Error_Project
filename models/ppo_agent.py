import os
from pathlib import Path
import yaml
import torch
import numpy as np
from models.ppo_model import ActorNetwork, CriticNetwork
from models.memory_buffer import MemoryBuffer


class FSO_Agent:
    def __init__(self, hyperparameter_set, state_dim, action_dim):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' #use gpu if available 
        #finding "hyperparameters.yml" and will only work if folder setup is correct
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        config_path = project_root / "env_" / "config" / "hyperparameters.yml"
        with open(config_path, 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        # -- creating and assigning hyperparameter variables from file --
        self.hyperparameter_set = hyperparameter_set
        self.batch_size = hyperparameters['batch_size']
        self.alpha = hyperparameters['alpha']
        self.gamma = hyperparameters['gamma']
        self.gae_lambda = hyperparameters['gae_lambda']
        self.policy_clip = hyperparameters['policy_clip']
        self.n_epochs = hyperparameters['n_epochs']
        
        self.actor = ActorNetwork(state_dim, action_dim, self.alpha)
        self.critic = CriticNetwork(state_dim, self.alpha)
        self.memory = MemoryBuffer(self.batch_size)
        
        self.actor.to(self.device)
        self.critic.to(self.device)

    @torch.no_grad()
    def select_action(self, observation):
        state = torch.tensor(np.array([observation]), dtype=torch.float32).to(self.device)
        
        #get action distribution from actor network then sample an action from it
        dist = self.actor(state)
        action = dist.sample()
        
        #calculate log probability of the action and value of the state for training
        log_prob = dist.log_prob(action).sum(axis=-1)
        value = self.critic(state)
        
        #flatten tensors to numpy values for the environment step
        action = action.cpu().detach().numpy()[0]
        log_prob = log_prob.cpu().detach().item()
        value = value.cpu().detach().item()
        
        return action, log_prob, value

    def optimize(self):
        if not self.memory.states:
            return
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, terminal_arr, _ = self.memory.generate_batches()
        if not all(np.all(np.isfinite(array)) for array in
                   (state_arr, action_arr, old_prob_arr, vals_arr, reward_arr)):
            raise ValueError("PPO rollout contains non-finite values")

        # Finite-horizon episodes include their final reward and end with V=0.
        # Compute returns BEFORE normalizing policy advantages.
        advantages = np.zeros(len(reward_arr), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(reward_arr))):
            next_value = vals_arr[t + 1] if t + 1 < len(vals_arr) else 0.0
            continuation = 1.0 - float(terminal_arr[t])
            delta = reward_arr[t] + self.gamma * next_value * continuation - vals_arr[t]
            gae = delta + self.gamma * self.gae_lambda * continuation * gae
            advantages[t] = gae
        returns = torch.tensor(advantages + vals_arr, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, terminal_arr, batches = self.memory.generate_batches()

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float32).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float32).to(self.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.float32).to(self.device)
                
                dist = self.actor(states)
                critic_value = self.critic(states).squeeze(-1)
                
                new_probs = dist.log_prob(actions).sum(axis=-1)
                prob_ratio = torch.exp((new_probs - old_probs).clamp(-20.0, 20.0))
                
                #-- PPO objective function clipping --
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                #value loss minimization
                critic_loss = torch.nn.functional.smooth_l1_loss(critic_value, returns[batch])
                
                #backpropagation step
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
                    raise FloatingPointError("Non-finite PPO loss; update stopped before changing weights")
                actor_loss.backward()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5, error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5, error_if_nonfinite=True)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
        self.memory.clear_memory()
