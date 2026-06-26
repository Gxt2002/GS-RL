import argparse
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def load_infos(input_path: Path):
    with input_path.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "infos" in data:
        return list(sorted(data["infos"], key=lambda e: e["timestamp"]))

    if isinstance(data, dict):
        infos = list(data.values())
        return list(sorted(infos, key=lambda e: e.get("timestamp", 0)))

    if isinstance(data, list):
        return list(sorted(data, key=lambda e: e.get("timestamp", 0)))

    raise TypeError(f"Unsupported input pickle format: {type(data)!r}")


def collect_trajectories(infos, future_steps: int, num_commands: int):
    navi_trajs = [[] for _ in range(num_commands)]

    for info in tqdm(infos, desc="Collecting trajectories"):
        if not all(k in info for k in ("gt_ego_fut_trajs", "gt_ego_fut_masks", "gt_ego_fut_cmd")):
            continue

        plan_traj = np.asarray(info["gt_ego_fut_trajs"], dtype=np.float32)
        plan_mask = np.asarray(info["gt_ego_fut_masks"])
        cmd = np.asarray(info["gt_ego_fut_cmd"], dtype=np.int32)

        if plan_traj.shape[0] < future_steps:
            continue

        plan_traj = plan_traj[:future_steps].cumsum(axis=-2)
        plan_mask = plan_mask[:future_steps]

        if int(plan_mask.sum()) != future_steps:
            continue

        cmd_idx = int(cmd.argmax(axis=-1))
        if cmd_idx < 0 or cmd_idx >= num_commands:
            continue

        navi_trajs[cmd_idx].append(plan_traj)

    return navi_trajs


def fit_clusters(navi_trajs, clusters_per_command: int, future_steps: int, random_state: int):
    clusters = []

    for cmd_idx, trajs in enumerate(navi_trajs):
        if len(trajs) < clusters_per_command:
            raise ValueError(
                f"Command {cmd_idx} has only {len(trajs)} valid trajectories, "
                f"less than n_clusters={clusters_per_command}."
            )

        trajs = np.stack(trajs, axis=0).reshape(-1, future_steps * 2)
        cluster = KMeans(
            n_clusters=clusters_per_command,
            random_state=random_state,
            n_init=10,
        ).fit(trajs).cluster_centers_
        clusters.append(cluster.reshape(-1, future_steps, 2).astype(np.float32))

    return np.stack(clusters, axis=0)


def save_preview(clusters: np.ndarray, output_png: Path):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    for cmd_clusters in clusters:
        for cluster in cmd_clusters:
            plt.scatter(cluster[:, 0], cluster[:, 1], s=8)
            plt.plot(cluster[:, 0], cluster[:, 1], linewidth=1)
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(output_png, bbox_inches="tight", dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plan_recon_K.npy trajectory anchors from VAD nuScenes info."
    )
    parser.add_argument(
        "--input",
        default="assets/nus/information/token2vad.pkl",
        help="Input pickle. Supports token2vad.pkl or vad_nuscenes_infos_temporal_train.pkl format.",
    )
    parser.add_argument(
        "--output-dir",
        default="assets/nus/kmeans",
        help="Directory for plan_recon_K.npy and preview image.",
    )
    parser.add_argument("--clusters", type=int, default=6, help="K clusters per command.")
    parser.add_argument("--future-steps", type=int, default=6, help="Number of future trajectory points.")
    parser.add_argument("--num-commands", type=int, default=3, help="Number of navigation commands.")
    parser.add_argument("--random-state", type=int, default=0, help="KMeans random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    infos = load_infos(input_path)
    navi_trajs = collect_trajectories(infos, args.future_steps, args.num_commands)

    for cmd_idx, trajs in enumerate(navi_trajs):
        print(f"command {cmd_idx}: {len(trajs)} valid trajectories")

    clusters = fit_clusters(navi_trajs, args.clusters, args.future_steps, args.random_state)

    output_npy = output_dir / f"plan_recon_{args.clusters}.npy"
    output_png = output_dir / f"plan_recon_{args.clusters}.png"

    np.save(output_npy, clusters)
    save_preview(clusters, output_png)

    print(f"saved {output_npy} with shape {clusters.shape}")
    print(f"saved preview {output_png}")


if __name__ == "__main__":
    main()
