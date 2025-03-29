import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque

class GaussianActor(nn.Module):
    def __init__(self, in_dim, hidden_dim, action_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        mean = self.mean(x)
        std = torch.exp(self.log_std).clamp(min=1e-3, max=1.0)  # 안정화
        return mean, std


class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.value(x)


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        mean, std = self.actor(state)
        value = self.critic(state)
        return mean, std, value

    def calculate_returns(self, rewards, gamma):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        return returns

    def calculate_advantages(self, returns, values):
        advantages = returns - values
        adv_std = advantages.std()
        if adv_std > 1e-6:
            advantages = (advantages - advantages.mean()) / adv_std
        return advantages

    def calculate_surrogate_loss(self, old_log_probs, new_log_probs, epsilon, advantages):
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        return torch.min(surr1, surr2)

    def calculate_losses(self, surrogate_loss, entropy, entropy_coef, returns, values):
        policy_loss = -torch.mean(surrogate_loss + entropy_coef * entropy)
        value_loss = F.smooth_l1_loss(values, returns).sum()
        return policy_loss, value_loss


class Agent:
    def __init__(self, env, model, optimizer, gamma=0.99, epsilon=0.2, entropy_coef=0.01):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef

    def run_episode(self):
        state, _ = self.env.reset()
        done = False
        episode_reward = 0

        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, std, value = self.model(state_tensor)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            # 안정적인 클리핑 처리
            action = action.squeeze(0)
            low = torch.tensor(self.env.action_space.low, dtype=action.dtype, device=action.device)
            high = torch.tensor(self.env.action_space.high, dtype=action.dtype, device=action.device)
            clipped_action = torch.clamp(action, low, high).detach().cpu().numpy()

            next_state, reward, terminated, truncated, _ = self.env.step(clipped_action)
            done = terminated or truncated

            states.append(state_tensor)
            actions.append(action.unsqueeze(0))
            log_probs.append(log_prob)
            values.append(value.squeeze(0))
            rewards.append(reward)
            episode_reward += reward
            state = next_state

        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        returns = self.model.calculate_returns(rewards, self.gamma)
        advantages = self.model.calculate_advantages(returns, values)

        return episode_reward, states, actions, log_probs, returns, advantages

    def update(self, states, actions, old_log_probs, returns, advantages):
        mean, std, values = self.model(states)
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        surrogate = self.model.calculate_surrogate_loss(
            old_log_probs, new_log_probs, self.epsilon, advantages
        )

        policy_loss, value_loss = self.model.calculate_losses(
            surrogate, entropy, self.entropy_coef, returns, values.squeeze(-1)
        )

        self.optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item()

def train(agent, num_episodes):
    best_score = -float('inf')
    reward_history = deque(maxlen=100)

    for episode in range(num_episodes):
        reward, states, actions, log_probs, returns, advantages = agent.run_episode()
        p_loss, v_loss = agent.update(states, actions, log_probs, returns, advantages)

        reward_history.append(reward)
        avg_reward = sum(reward_history) / len(reward_history)

        score = avg_reward - (p_loss * 0.5) - (v_loss * 0.5)

        if score > best_score:
            best_score = score
            torch.save(agent.model.state_dict(), 'best_model.pth')
            print(f"[Saved best model] Episode {episode} | Score: {score:.2f} | Avg Reward: {avg_reward:.2f} | Policy Loss: {p_loss:.4f} | Value Loss: {v_loss:.4f}")

        if episode % 1000 == 0:
            print(f"Episode {episode} | Reward: {reward:.2f} | Avg Reward: {avg_reward:.2f} | Policy Loss: {p_loss:.4f} | Value Loss: {v_loss:.4f}")

def create_networks(env):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    actor = GaussianActor(obs_dim, hidden_dim=256, action_dim=act_dim)
    critic = Critic(obs_dim, hidden_dim=256)
    model = ActorCritic(actor, critic)
    return model
