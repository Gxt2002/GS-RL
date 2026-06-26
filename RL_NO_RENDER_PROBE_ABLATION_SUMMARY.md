# GSDrive RL No-Render Probe 消融实验总结

## 背景

在 `RL_MULTIGPU_OPTIMIZATION_PLAN.md` 中，我们判断 RL 阶段的主要瓶颈来自 trajectory probe：probe 只是为了评估候选未来轨迹的 reward，不会把 probe 中生成的 RGB 图像喂给策略网络；但原始实现调用完整 `env.step()`，每个 probe step 都会渲染 6 路 3DGS 相机图像。

因此本次阶段 1 优化的目标是：保留 trajectory probe reward 和真实 rollout 的 3DGS 闭环渲染，只在 probe-only 未来试探中跳过 RGB 渲染。

## 改动概要

- `reconsimulator/envs/nus.py`：新增 `step_no_render(action)`，复用 ego pose 更新、碰撞和 info 逻辑，但不调用 `_get_obs()`、不调用 `get_sky_view()`。
- `env.py`：新增 `step_reward_only(action)`，抽出 `_compute_step_reward()`，让 full step 和 no-render step 使用同一套 reward 计算。
- `model_infra/rewarding.py`：`TrajectoryProbeReward` 优先调用 `step_reward_only()`，保留 full-render fallback。
- `train.py`：新增 `use_no_render_probe`、`perf_log_path`，并记录 `probe_s`、`env_step_render_s`、`trajectory_mode_sample_s`、`ppo_update_s` 等 timing。
- `train_rl_stage2.py`：新增只跑 RL 阶段的脚本，默认超参对齐原始 RL run，并支持 `--full-render-probe` 做严格对照。

真实 rollout 没有改：策略实际执行动作后仍然调用完整 `env.step()`，仍然渲染 3DGS 图像作为下一步 observation。

## 消融设置

两组实验使用同一组超参：

```text
num_envs=8
num_steps=8
rollout_batch=64
minibatch_size=4
update_epochs=2
total_timesteps=32000
learning_rate=1e-5
use_trajectory_probe=True
use_traj_probe=True
seed=0
```

唯一差异：

```text
no-render:   use_no_render_probe=True
full-render: use_no_render_probe=False
```

## 结果对齐

前两个 PPO update 中，no-render 与 full-render 的训练指标几乎逐项一致。

```text
Update 1:
reward=2.305±0.195, return=8.737, SR=1.000, collision=0.000,
deviation=0.000, traj_probe=1.358, loss=-0.8075

Update 2:
reward=2.278±0.255, return=10.417, SR=1.000, collision=0.000,
deviation=0.000, traj_probe=1.371, loss=-0.8542
```

PPO loss 序列也只有极小浮点差异，说明 no-render probe 没有改变训练语义。

## 耗时对比

单轮耗时：

```text
Update 1:
no-render   total=212.76s,  speed=16.92 upd/h, probe=4.96s
full-render total=1464.45s, speed=2.46 upd/h,  probe=1252.90s

Update 2:
no-render   total=229.98s,  speed=16.26 upd/h, probe=5.21s
full-render total=1447.99s, speed=2.47 upd/h,  probe=1239.72s
```

平均耗时：

```text
no-render 前 2 个 update:
update_total_s = 221.37s
probe_s = 5.09s
env_step_render_s = 59.29s
trajectory_mode_sample_s = 123.85s
ppo_update_s = 21.46s

full-render 前 3 个 update:
update_total_s = 1467.60s
probe_s = 1256.97s
env_step_render_s = 59.63s
trajectory_mode_sample_s = 119.50s
ppo_update_s = 20.33s
```

提速倍数：

```text
按平均 update 总耗时: 1467.60 / 221.37 = 6.63x
按前两轮速度均值:    16.59 / 2.47 ≈ 6.72x
```

## 完整耗时分解

下面所有百分比都以 `update_total_s` 为分母。`rollout_s` 是总项，内部包含 `probe_s`、`trajectory_mode_sample_s`、真实 `env.step()` 等子项，所以不要把 `rollout_s` 和其子项相加。看瓶颈时优先看子项的秒数和占比。

No-render 前 2 个 update 平均：

```text
update_total_s                  221.37s   100.0%
rollout_s                       198.55s    89.7%   # rollout 总耗时，包含下面多个子项
  trajectory_mode_sample_s      123.85s    56.0%   # 当前最大瓶颈
  env_step_render_s              59.29s    26.8%   # 真实 rollout 的 3DGS 渲染
  model_forward_s                10.23s     4.6%
  probe_s                         5.09s     2.3%   # no-render 后已不是瓶颈
gae_metrics_s                     1.35s     0.6%
ppo_update_s                     21.46s     9.7%
unattributed/overhead             0.00s     0.0%
```

Full-render 前 3 个 update 平均：

```text
update_total_s                 1467.60s   100.0%
rollout_s                      1446.03s    98.5%   # rollout 总耗时，包含下面多个子项
  probe_s                      1256.97s    85.7%   # 原始最大瓶颈
  trajectory_mode_sample_s      119.50s     8.1%
  env_step_render_s              59.63s     4.1%
  model_forward_s                 9.85s     0.7%
gae_metrics_s                     1.24s     0.1%
ppo_update_s                     20.33s     1.4%
unattributed/overhead             0.00s     0.0%
```

关键变化是：probe 耗时从约 `1257s/update` 降到约 `5s/update`。no-render 后 probe 不再是主瓶颈，新的瓶颈排序变为：

```text
1. trajectory_mode_sample_s 约 124s/update，约 56.0%
2. env_step_render_s       约  59s/update，约 26.8%
3. ppo_update_s            约  21s/update，约  9.7%
4. model_forward_s         约  10s/update，约  4.6%
5. probe_s                 约   5s/update，约  2.3%
```

## 结论

本次 no-render probe 优化验证了计划文档中的阶段 1 判断：

- probe 只需要未来 reward，不需要 RGB observation。
- 真实闭环视觉仿真保持不变。
- 严格同超参对照下，训练指标与 full-render probe 对齐。
- 单卡 RL 训练速度提升约 `6.6x-6.7x`。

因此，该改动可以作为后续多卡同步 rollout/probe 的基础版本。下一步应优先并行化真实 rollout/render 和 trajectory mode sampling，而不是减少 `traj_probe_num_modes` 这种会改变训练信号的配置。

