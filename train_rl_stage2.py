import argparse
import os
import random

import numpy as np
import torch
from train import discover_available_scenes, train_ppo


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run GSDrive stage-2 PPO/RL training only.")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scene", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--updates", type=int, default=5, help="Short-run PPO updates for validation.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--init-model-path", type=str, default="bc_reconsimulator_v2.pt")
    parser.add_argument("--save-path", type=str, default="ppo_reconsimulator_v2.pt")
    parser.add_argument("--perf-log-path", type=str, default="rl_perf_logs_stage2.jsonl")
    parser.add_argument("--tensorboard-log-dir", type=str, default="runs/gsdrive")
    parser.add_argument("--full-render-probe", action="store_true", help="Disable no-render probe for A/B comparison.")
    parser.add_argument("--disable-traj-probe", action="store_true")
    parser.add_argument("--disable-trajectory-probe", action="store_true")
    parser.add_argument("--traj-probe-num-modes", type=int, default=6)
    parser.add_argument("--probe-num-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    scenes = discover_available_scenes(default_scene=args.scene)
    if args.scene is not None and args.scene in scenes:
        scenes = [args.scene] + [scene for scene in scenes if scene != args.scene]

    total_timesteps = args.total_timesteps
    if total_timesteps is None:
        total_timesteps = args.updates * args.num_envs * args.num_steps

    if not os.path.exists(args.init_model_path):
        raise FileNotFoundError(
            f"BC checkpoint not found: {args.init_model_path}. "
            "Pass --init-model-path to the completed imitation-learning checkpoint."
        )

    print("=" * 80)
    print("Stage-2 RL/PPO Training")
    print("=" * 80)
    print(
        f"updates~{args.updates}, total_timesteps={total_timesteps}, "
        f"num_envs={args.num_envs}, num_steps={args.num_steps}, "
        f"minibatch_size={args.minibatch_size}, "
        f"use_no_render_probe={not args.full_render_probe}, "
        f"perf_log_path={args.perf_log_path}"
    )

    train_ppo(
        cuda=args.cuda,
        scene=scenes[0],
        total_timesteps=total_timesteps,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        num_envs=args.num_envs,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        device=args.device,
        save_path=args.save_path,
        init_model_path=args.init_model_path,
        scenes=scenes,
        use_trajectory_probe=not args.disable_trajectory_probe,
        probe_num_steps=args.probe_num_steps,
        use_traj_probe=not args.disable_traj_probe,
        traj_probe_num_modes=args.traj_probe_num_modes,
        use_no_render_probe=not args.full_render_probe,
        perf_log_path=args.perf_log_path,
        tensorboard_log_dir=args.tensorboard_log_dir,
    )


if __name__ == "__main__":
    main()
