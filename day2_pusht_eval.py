"""Day 2 — watch a pretrained Diffusion Policy play PushT and save the video."""
from pathlib import Path
import gymnasium as gym
import gym_pusht  # noqa: F401  (registers the PushT env)
import imageio
import numpy as np
import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")
policy.to(device)
policy.reset()

import sys
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
obs, info = env.reset(seed=seed)
print(f"seed: {seed}")

frames = [env.render()]
done = False
step = 0
while not done and step < 300:
    state = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device)
    image = torch.from_numpy(obs["pixels"]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    with torch.inference_mode():
        action = policy.select_action({"observation.state": state, "observation.image": image})
    obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
    frames.append(env.render())
    done = terminated or truncated
    step += 1

out = Path.home() / "robotics-lab" / f"day2_pusht_seed{seed}.mp4"
imageio.mimsave(str(out), np.stack(frames), fps=25)
print(f"steps: {step} | success: {terminated} | video: {out}")
