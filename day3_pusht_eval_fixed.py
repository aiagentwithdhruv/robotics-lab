"""Day 3 — the frozen-agent fix.

Day 2 bug: LeRobot 0.6.0 removed normalization from the policy; the old
lerobot/diffusion_pusht checkpoint's stats loaded as "unexpected keys" and were
silently dropped -> garbage-scaled inputs -> agent never moved.

Fix: ran LeRobot's official migration (lerobot.processor.migrate_policy_normalization)
which extracts the stats into pre/post processor pipelines. This script evals the
migrated checkpoint with those processors in the loop.
"""
import sys
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers the PushT env)
import imageio
import numpy as np
import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

CKPT = Path.home() / "robotics-lab" / "checkpoints" / "diffusion_pusht_migrated"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

policy = DiffusionPolicy.from_pretrained(str(CKPT))
policy.to(device)
policy.reset()

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=str(CKPT),
    preprocessor_overrides={"device_processor": {"device": device}},
)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
obs, info = env.reset(seed=seed)
print(f"seed: {seed}")

frames = [env.render()]
done = False
step = 0
while not done and step < 300:
    state = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0)
    image = torch.from_numpy(obs["pixels"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    batch = preprocessor({"observation.state": state, "observation.image": image})
    with torch.inference_mode():
        action = policy.select_action(batch)
    action = postprocessor(action)
    obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
    frames.append(env.render())
    done = terminated or truncated
    step += 1

out = Path.home() / "robotics-lab" / f"day3_pusht_fixed_seed{seed}.mp4"
imageio.mimsave(str(out), np.stack(frames), fps=25)
print(f"steps: {step} | success: {terminated} | video: {out}")
