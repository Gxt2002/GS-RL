import argparse
import json
import random

import numpy as np

from env import ReconNusPPOEnv


def compare_action(env, action):
    base_state = env.save_state()

    _, full_reward, full_terminated, full_truncated, full_info = env.step(action)
    full_ego = env.base_env.start_ego.copy()

    env.restore_state(base_state)

    reward_only_reward, reward_only_terminated, reward_only_truncated, reward_only_info = env.step_reward_only(action)
    reward_only_ego = env.base_env.start_ego.copy()

    env.restore_state(base_state)

    def info_float(info, key, default=0.0):
        try:
            return float(info.get(key, default))
        except Exception:
            return float(default)

    return {
        "action": [int(action[0]), int(action[1])],
        "reward_full": float(full_reward),
        "reward_no_render": float(reward_only_reward),
        "reward_abs_diff": float(abs(full_reward - reward_only_reward)),
        "terminated_match": bool(full_terminated == reward_only_terminated),
        "truncated_match": bool(full_truncated == reward_only_truncated),
        "ego_max_abs_diff": float(np.max(np.abs(full_ego - reward_only_ego))),
        "distance_abs_diff": float(abs(info_float(full_info, "distance") - info_float(reward_only_info, "distance"))),
        "yaw_v_abs_diff": float(abs(info_float(full_info, "yaw_v") - info_float(reward_only_info, "yaw_v"))),
        "ego2match_yaw_abs_diff": float(
            abs(info_float(full_info, "ego2match_yaw_degrees") - info_float(reward_only_info, "ego2match_yaw_degrees"))
        ),
        "collision_full": full_info.get("collision"),
        "collision_no_render": reward_only_info.get("collision"),
        "probe_step_mode": reward_only_info.get("probe_step_mode"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare full-render env.step with no-render reward-only step.")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--num-actions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="no_render_probe_validation.json")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = ReconNusPPOEnv(cuda=args.cuda, scene=args.scene, debug=False, resize_shape=(112, 200))
    env.reset(seed=args.seed)

    nvec = env.action_space.nvec
    results = []
    for _ in range(args.num_actions):
        action = [
            random.randrange(int(nvec[0])),
            random.randrange(int(nvec[1])),
        ]
        result = compare_action(env, action)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    summary = {
        "scene": int(args.scene),
        "num_actions": int(args.num_actions),
        "max_reward_abs_diff": max((r["reward_abs_diff"] for r in results), default=0.0),
        "max_ego_abs_diff": max((r["ego_max_abs_diff"] for r in results), default=0.0),
        "max_distance_abs_diff": max((r["distance_abs_diff"] for r in results), default=0.0),
        "all_terminated_match": all(r["terminated_match"] for r in results),
        "all_truncated_match": all(r["truncated_match"] for r in results),
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("summary:", json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False))
    print(f"saved validation report to {args.output}")


if __name__ == "__main__":
    main()
