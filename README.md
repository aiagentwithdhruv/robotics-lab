# Robotics Lab 🦾 — learning robotics in public

Sim-first robot learning on a MacBook Air (M5, MPS) with [LeRobot](https://github.com/huggingface/lerobot). No robotics background — building the skills from zero, one 30-minute session a day, everything logged.

**The thesis:** the bottleneck in robotics is the intelligence, not the hardware. I build AI agent systems for a living; this lab is where that brain-building meets bodies.

## Log
| Day | Date | What happened |
|-----|------|---------------|
| 1 | Jul 17 2026 | LeRobot installed, MPS verified on Apple Silicon |
| 2 | Jul 18 2026 | HF Course Unit 0 · pretrained Diffusion Policy loaded on MPS, PushT env running · 6 real bugs debugged (Python 3.14 incompat, missing extras, pymunk API break, ffmpeg backend) · honest ending: checkpoint↔library version mismatch left the agent frozen — normalization stats silently failed to load. Warnings are load-bearing. Fix = Day 3 |

## Setup
```bash
python3.12 -m venv ~/lerobot-env-312
source ~/lerobot-env-312/bin/activate
pip install lerobot gym-pusht imageio
```

Plan: 8 weeks sim-only (HF Robotics Course → train ACT/Diffusion on PushT) → then real hardware (SO-101 arm).
