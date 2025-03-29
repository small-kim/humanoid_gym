import gymnasium as gym
import torch
import numpy as np
from agent import Agent, create_networks, train
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
from pathlib import Path

# 이동 평균 함수 정의
def moving_average(data, window_size=10):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():
    env = gym.make("Humanoid-v5", render_mode=None)  # GUI 없이 실행
    env.reset(seed=42)
    torch.manual_seed(42)
    np.random.seed(42)

    model = create_networks(env)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    agent = Agent(env, model, optimizer, gamma=0.99, epsilon=0.2, entropy_coef=0.01)

    reward_record = []
    loss_record = []
    critic_record = []
    entropy_record = []

    save_dir = Path("plot_trend")  # 현재 디렉토리 내부로 경로 설정
    save_dir.mkdir(parents=True, exist_ok=True)

    num_episodes = 1_000_000
    best_score = -float('inf')

    for ep in range(num_episodes):
        reward, states, actions, log_probs, returns, advantages = agent.run_episode()
        policy_loss, value_loss = agent.update(states, actions, log_probs, returns, advantages)

        reward_record.append(reward)
        loss_record.append(policy_loss + value_loss)
        critic_record.append(value_loss)
        entropy_record.append(policy_loss)  # Entropy 따로 계산하려면 agent에서 반환 필요

        score = reward - 0.5 * (policy_loss + value_loss)
        if score > best_score:
            best_score = score
            torch.save(agent.model.state_dict(), 'best_model.pth')
            print(f"best_model.pth 저장됨 (에피소드 {ep}, score: {score:.2f})")

        # 100번째마다: 최근 100개만 그래프를 저장함
        if ep % 100 == 0 and ep > 0:
            recent_rewards = reward_record[-100:]
            recent_losses = loss_record[-100:]
            recent_critics = critic_record[-100:]
            recent_entropies = entropy_record[-100:]

            recent_avg = np.mean(recent_rewards)
            clear_output(True)
            print(f'{ep}번째 에피소드 결과')
            print(f'최근 {ep - 100} - {ep} 에피소드 보상평균 = {recent_avg:.2f}')

            plt.figure(figsize=[12, 12])

            plt.subplot(2,2,1)
            plt.title("Total Reward")
            plt.plot(recent_rewards, label="reward")
            plt.plot(moving_average(recent_rewards), label="avg")
            plt.grid()
            plt.legend()

            plt.subplot(2,2,2)
            plt.title("Loss trend")
            plt.plot(recent_losses, label="loss")
            plt.plot(moving_average(recent_losses), label="avg")
            plt.grid()
            plt.legend()

            plt.subplot(2,2,3)
            plt.title("Advantage trend")
            plt.plot(recent_critics, label="value loss")
            plt.plot(moving_average(recent_critics), label="avg")
            plt.grid()
            plt.legend()

            plt.subplot(2,2,4)
            plt.title("Entropy trend")
            plt.plot(recent_entropies, label="policy loss")
            plt.plot(moving_average(recent_entropies), label="avg")
            plt.grid()
            plt.legend()

            plot_path = save_dir / "every_100" / f'plot_ep{ep}.png'
            plt.savefig(plot_path)
            plt.close()

        # 1000번째마다: 최근 1000개만 그래프를 저장함
        if ep % 1_000 == 0 and ep > 0:
            recent_rewards = reward_record[-1_000:]
            recent_losses = loss_record[-1_000:]
            recent_critics = critic_record[-1_000:]
            recent_entropies = entropy_record[-1_000:]

            recent_avg = np.mean(recent_rewards)
            clear_output(True)
            print(f'{ep}번째 에피소드 결과')
            print(f'최근 {ep - 1000} - {ep} 에피소드 보상평균 = {recent_avg:.2f}')

            plt.figure(figsize=[12, 12])

            plt.subplot(2,2,1)
            plt.title("Total Reward")
            plt.plot(recent_rewards, label="reward")
            plt.plot(moving_average(recent_rewards), label="avg")
            plt.grid()
            plt.legend()

            plt.subplot(2,2,2)
            plt.title("Loss trend")
            plt.plot(recent_losses, label="loss")
            plt.plot(moving_average(recent_losses), label="avg")
            plt.grid()
            plt.legend()

            plt.subplot(2,2,3)
            plt.title("Advantage trend")
            plt.plot(recent_critics, label="value loss")
            plt.plot(moving_average(recent_critics), label="avg")
            plt.grid()
            plt.legend()

            plt.subplot(2,2,4)
            plt.title("Entropy trend")
            plt.plot(recent_entropies, label="policy loss")
            plt.plot(moving_average(recent_entropies), label="avg")
            plt.grid()
            plt.legend()

            plot_path = save_dir / "every_1_000"/ f'plot_ep{ep}_every_thousand.png'
            plt.savefig(plot_path)
            plt.close()

        # 1_0000번째마다: 전체 그래프를 저장함
        if ep % 10_000 == 0 and ep > 0:
            total_avg = np.mean(reward_record)
            clear_output(True)
            print(f'{ep}번째 에피소드 결과')
            print(f'전체 에피소드 보상평균 = {total_avg:.2f}')

            plt.figure(figsize=[12, 12])

            plt.subplot(2,2,1)
            plt.title("Total Reward")
            plt.plot(reward_record, label="reward")
            plt.plot(moving_average(reward_record), label="avg")
            plt.grid()
            plt.legend()

            plt.subplot(2,2,2)
            plt.title("Loss trend")
            plt.plot(loss_record, label="loss")
            plt.plot(moving_average(loss_record), label="avg")
            plt.grid()
            plt.legend()

            plt.subplot(2,2,3)
            plt.title("Advantage trend")
            plt.plot(critic_record, label="value loss")
            plt.plot(moving_average(critic_record), label="avg")
            plt.grid()
            plt.legend()

            plt.subplot(2,2,4)
            plt.title("Entropy trend")
            plt.plot(entropy_record, label="policy loss")
            plt.plot(moving_average(entropy_record), label="avg")
            plt.grid()
            plt.legend()

            plot_path = save_dir / "every_10_000_total" / f'plot_ep{ep}_total.png'
            plt.savefig(plot_path)
            plt.close()

    env.close()

if __name__ == "__main__":
    main()