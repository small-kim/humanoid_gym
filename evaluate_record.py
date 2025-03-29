import gymnasium as gym
import torch
import numpy as np
from agent import create_networks
from gymnasium.wrappers import RecordVideo
from pathlib import Path
import imageio


def evaluate_with_record(
    model_path="best_model.pth",
    save_dir="best_model_video",
    episodes=3,
    save_frames=True
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for ep in range(episodes):
        video_subdir = save_path / f"episode_{ep+1}"
        video_subdir.mkdir(parents=True, exist_ok=True)

        env = gym.make("Humanoid-v5", render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_folder=str(video_subdir),
            episode_trigger=lambda x: True,
            name_prefix=f"humanoid_ep{ep+1}"
        )

        env.reset(seed=42 + ep)
        torch.manual_seed(42 + ep)
        np.random.seed(42 + ep)

        model = create_networks(env)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        state, _ = env.reset()
        done = False
        total_reward = 0
        frame_list = []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                mean, std, _ = model(state_tensor)
                action = mean.squeeze(0)
                low = torch.tensor(env.action_space.low, dtype=action.dtype)
                high = torch.tensor(env.action_space.high, dtype=action.dtype)
                clipped_action = torch.clamp(action, low, high).detach().cpu().numpy()

            next_state, reward, terminated, truncated, _ = env.step(clipped_action)
            frame = env.render()

            if save_frames:
                frame_list.append(frame)

            done = terminated or truncated
            total_reward += reward
            state = next_state

        print(f"[RECORDED] Episode {ep+1} | Reward: {total_reward:.2f}")

        if save_frames:
            gif_path = video_subdir / f"humanoid_ep{ep+1}.gif"
            imageio.mimsave(gif_path, frame_list, fps=30)
            print(f"🖼️ Saved GIF: {gif_path}")

        env.close()

    print(f"✅ All videos saved to: {save_path.resolve()}")


if __name__ == "__main__":
    evaluate_with_record(episodes=3, save_frames=True)
