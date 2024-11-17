# src/models/learning.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque
import random
from dataclasses import dataclass
import logging

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class LearningModule:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks and buffers
        self.policy_net = PolicyNetwork(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            hidden_dim=config['hidden_dim']
        ).to(self.device)
        
        self.target_net = PolicyNetwork(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            hidden_dim=config['hidden_dim']
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config['learning_rate']
        )
        
        self.memory = ReplayBuffer(config['memory_size'])
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']  # discount factor
        self.tau = config['tau']  # soft update parameter
        self.epsilon = config['epsilon_start']
        
        self.logger = logging.getLogger("LearningModule")

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.config['action_dim'])
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()

    def update(self, experience: Experience):
        """Update the learning module with new experience."""
        # Store experience in replay buffer
        self.memory.push(experience)
        
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch and perform update
        self._update_policy()
        
        # Update target network
        self._soft_update_target_network()
        
        # Update exploration rate
        self._update_epsilon()

    def _update_policy(self):
        """Perform one step of policy update."""
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        
        # Prepare batch data
        state_batch = torch.FloatTensor([e.state for e in batch]).to(self.device)
        action_batch = torch.LongTensor([e.action for e in batch]).to(self.device)
        reward_batch = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_state_batch = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        done_batch = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.logger.debug(f"Policy update - Loss: {loss.item():.4f}")

    def _soft_update_target_network(self):
        """Soft update target network parameters."""
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

    def _update_epsilon(self):
        """Update exploration rate."""
        self.epsilon = max(
            self.config['epsilon_end'],
            self.epsilon * self.config['epsilon_decay']
        )

    def extract_insights(self, experience_batch: List[Experience]) -> Dict:
        """Extract insights from recent experiences."""
        if len(experience_batch) == 0:
            return {}
        
        # Calculate average reward
        avg_reward = np.mean([e.reward for e in experience_batch])
        
        # Identify successful actions
        successful_actions = [
            e.action for e in experience_batch
            if e.reward > self.config['success_threshold']
        ]
        
        # Calculate action effectiveness
        action_effectiveness = {}
        for action in range(self.config['action_dim']):
            action_experiences = [
                e for e in experience_batch
                if e.action == action
            ]
            if action_experiences:
                avg_action_reward = np.mean([e.reward for e in action_experiences])
                action_effectiveness[action] = avg_action_reward
        
        return {
            'average_reward': avg_reward,
            'successful_actions': successful_actions,
            'action_effectiveness': action_effectiveness,
            'exploration_rate': self.epsilon
        }

    def save_model(self, path: str):
        """Save model parameters."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.logger.info(f"Model loaded from {path}")

# Example configuration for LearningModule
learning_config = {
    'state_dim': 10,          # Dimension of state space
    'action_dim': 4,          # Number of possible actions
    'hidden_dim': 64,         # Hidden layer size
    'learning_rate': 0.001,   # Learning rate
    'memory_size': 10000,     # Size of replay buffer
    'batch_size': 64,         # Batch size for updates
    'gamma': 0.99,            # Discount factor
    'tau': 0.005,            # Soft update parameter
    'epsilon_start': 1.0,     # Initial exploration rate
    'epsilon_end': 0.01,      # Final exploration rate
    'epsilon_decay': 0.995,   # Exploration rate decay
    'success_threshold': 0.7  # Threshold for successful actions
}

# Usage example:
if __name__ == "__main__":
    # Create learning module
    learning_module = LearningModule(learning_config)
    
    # Simulate some experiences
    for _ in range(100):
        state = np.random.random(learning_config['state_dim'])
        action = learning_module.select_action(state)
        reward = random.random()
        next_state = np.random.random(learning_config['state_dim'])
        done = random.random() > 0.9
        
        experience = Experience(state, action, reward, next_state, done)
        learning_module.update(experience)
    
    # Extract insights
    insights = learning_module.extract_insights([
        Experience(
            np.random.random(learning_config['state_dim']),
            random.randrange(learning_config['action_dim']),
            random.random(),
            np.random.random(learning_config['state_dim']),
            random.random() > 0.9
        )
        for _ in range(10)
    ])
    
    print("Learning Insights:", insights)