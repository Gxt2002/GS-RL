# GSDrive RL 多卡优化改造计划

## 目标

在尽量保持原始训练语义和效果的前提下，提高 RL 阶段训练吞吐：

- 保留 trajectory probe reward，因为这是这个项目的重要亮点。
- 默认保持 PPO 的 on-policy 训练方式。
- 不随意改 reward 定义，除非有明确验证。
- 使用多卡并行加速最昂贵的 rollout 和 probe 流程。
- 减少 probe-only 未来试探过程里不必要的 3DGS 渲染。

从当前日志看，主要瓶颈是：

- `speed ~= 2.6 upd/h`
- `num_envs=8`，`num_steps=16`，所以每个 PPO update 会收集 `128` 个真实 rollout step。
- 两套 probe 都打开时，每个真实 step 会触发大量额外的 probe 环境步。
- 当前 probe 调的是完整 `env.step()`，即使 probe 只需要 reward 和指标，也会渲染 6 路相机图像。

## 当前训练流程

当前 PPO 是同步、单进程流程：

1. learner 模型根据所有 env 的当前观测采样动作。
2. 每个 rollout step 中，代码可能会额外采样多条 trajectory mode。
3. 对每个 env、每条采样轨迹，trajectory probe 调用 `compute_future_reward()`。
4. `compute_future_reward()` 调用 `probe_environment()`。
5. `probe_environment()` 内部反复调用完整 `env.step()`。
6. 完整 `env.step()` 会更新 ego pose，通过 3DGS 渲染 6 路相机图像，构造观测张量，并计算 reward/info。
7. 收集满 `num_envs * num_steps` 条样本后，PPO 计算 GAE，并用少量 epoch 更新模型。
8. 这批旧 rollout 数据被丢弃，下一轮重新用新策略采样。

当前实现符合 on-policy PPO 的基本流程，但没有跨 GPU 并行。`cuda` 参数只传入一个值，所以所有 simulator 实例都会放在同一张 GPU 上。

## 主要瓶颈

1. Probe 执行了完整渲染。

   Probe 只用于未来 reward 评估。完整 `env.step()` 返回的 RGB observation 在 probe 阶段不会喂给策略网络。因此每个 probe step 都渲染 6 路相机图像是浪费。

2. Probe 循环是串行的。

   当前代码按 Python for-loop 依次遍历 env、mode、probe step。即使机器有多张 GPU，当前结构也不会把这些独立的 probe 任务并发分发出去。

3. 所有 env 都使用同一张 GPU。

   `make_env(cuda=cuda, ...)` 对所有 env 使用同一个 GPU id，导致 3DGS 渲染和 probe 仿真无法自然利用多卡。

4. PPO minibatch 太小。

   当前 `minibatch_size=4` 会产生很多很小的 forward/backward，调度开销高，GPU 利用率通常也不好。

5. Debug 设置有额外开销。

   `torch.autograd.set_detect_anomaly(True)` 对排查 NaN/反传异常很有用，但稳定训练时会明显拖慢。

6. Episode 终止条件需要检查。

   simulator 每次让 `now_frame` 增加 `step_frames=5`，但当前终止逻辑有接近 equality 的判断。建议检查并在必要时改为 `now_frame >= final_frame`，避免 episode 没有按预期结束。

## 阶段 1：No-Render Probe

这是第一优先级，也是性价比最高的改动。它保留 probe 思想，只去掉 probe 阶段不必要的图像渲染。

### 设计

新增一条 reward-only 环境步路径：

- `ReconSimulator.step_no_render(action)`
- `ReconNusPPOEnv.step_reward_only(action)`

reward-only 路径需要：

1. 像 `step()` 一样更新 `now_frame`。
2. 像 `ReconSimulator.step()` 一样根据 action anchor 更新 ego pose。
3. 使用相同的 `updateGroundDistance()` 更新地面高度。
4. 使用相同的 `check_coliision()` 计算碰撞。
5. 使用和 `ReconNusPPOEnv.step()` 一致的逻辑计算 distance-to-expert、yaw、comfort、progress、alignment 和 reward。
6. 只返回 `reward`、`terminated`、`truncated` 和 `info`。
7. 不调用 `_get_obs()`。
8. 不调用 `get_sky_view()`。
9. 不构造 `all_camera_now`。
10. 不 resize 或拼接 RGB 图像。

然后修改 `TrajectoryProbeReward.probe_environment()`，优先使用：

```python
if hasattr(env, "step_reward_only"):
    reward, terminated, truncated, info = env.step_reward_only([ax, ay])
else:
    _, reward, terminated, truncated, info = env.step([ax, ay])
```

fallback 用于保持兼容性。

### 预期影响

这不应该改变策略真实看到的 observation，因为真实 rollout 仍然使用完整 `env.step()`，仍然渲染 3DGS 相机图像。

只有一种情况可能影响训练：如果某些 reward 或 collision 信号隐含依赖渲染图像。根据当前代码，reward 主要来自 ego pose、专家轨迹距离、碰撞检测、yaw、agents 和 comfort 指标，而不是 RGB 像素。因此预期对策略效果影响很小。

### 验证方式

新增一个小的等价性检查脚本或 debug 模式：

1. 保存 env 当前状态。
2. 对同一个 action 执行完整 `env.step()`。
3. 恢复 env 状态。
4. 对同一个 action 执行 `step_reward_only()`。
5. 对比：
   - `reward`
   - `terminated`
   - `truncated`
   - `distance`
   - `collision`
   - `ego2match_yaw_degrees`
   - `yaw_v`
   - 最终 `start_ego`

验收目标：

- pose/reward 相关字段的数值差异应接近 0。
- 只有 observation-only 字段允许不同。

## 阶段 2：单进程多卡设备分配

这是一个风险较低的中间步骤，但因为 Python 循环仍是串行的，它不能彻底解决扩展性问题。

### 设计

让 `train_ppo()` 支持 `device_ids`，例如：

```python
device_ids=[0, 1, 2, 3]
```

创建 env 时按 GPU 轮转：

```python
env_cuda = device_ids[i % len(device_ids)]
envs.append(make_env(cuda=env_cuda, scene=int(sc), debug=False, resize_shape=resize_shape))
```

learner 模型仍然放在 `main_device` 上，初始可以设为 `device_ids[0]`。

### 预期效果

这可能降低单卡显存压力，也可能让部分 CUDA kernel 有机会重叠执行。但由于 Python 仍然串行 step env，提速有限。

估计提速：

- 保守：`1.2x-2x`
- 如果 CUDA kernel 重叠较好：`2x-3x`

这个阶段主要是兼容性检查和过渡，不是最终多卡方案。

## 阶段 3：同步多卡 Rollout Workers

这是推荐的多卡实现方式，因为它最能保持 PPO 语义。

### 为什么优先同步

PPO 是 on-policy。如果 actor 在 learner 更新时继续采样，样本会变成旧策略数据。旧策略数据会让 ratio 修正变大，可能影响 PPO 稳定性。

为了尽量保持原算法，应使用同步分布式 rollout：

```text
冻结 policy_k
把 policy_k 广播给所有 rollout worker
workers 并行采集 rollout/probe 数据
learner 汇总全部 rollout 数据
learner 执行 PPO update: policy_k -> policy_{k+1}
广播 policy_{k+1}
进入下一轮
```

这样可以保证每个 PPO batch 都来自同一个策略版本。

### 进程布局

如果有 16 张 GPU：

- 1 个 learner 进程放在一张 GPU 上。
- 15 或 16 个 rollout worker 进程。
- 每个 rollout worker 独占一张 GPU，并维护一个或多个 `ReconNusPPOEnv`。
- 每个 worker 加载一份 policy，用于 action sampling 和 trajectory mode sampling。
- learner 独占 optimizer，负责所有 PPO 梯度更新。

两种可行布局：

1. Learner + 15 actors：
   - GPU 0：learner，可选跑一个小 actor
   - GPU 1-15：rollout/probe actors

2. actor-learner 混合：
   - GPU 0 在显存允许时也跑 actor env
   - 16 张 GPU 都参与 rollout

### Worker 职责

每个 worker 应该：

1. 接收当前 policy version 的模型权重。
2. 采集 `local_num_envs * num_steps` 个 rollout 样本。
3. 保持 probe 开启。
4. probe-only 未来仿真使用 no-render probe。
5. 返回 rollout tensors：
   - observations
   - actions
   - old logprobs
   - values
   - rewards
   - dones
   - agents
   - target trajectories
   - camera intrinsics/extrinsics
   - metrics
   - probe diagnostics

learner 应该：

1. 拼接所有 worker 的 batch。
2. 计算 GAE/returns，或者接收 policy-version 对齐的预计算字段。
3. 执行 PPO update。
4. 保存 checkpoint 和 TensorBoard 日志。
5. 广播更新后的模型权重。

### 通信方案

初始实现建议：

- Python `torch.multiprocessing` + `spawn`。
- 用 `multiprocessing.Queue` 或 `torch.distributed` object collectives 传输 rollout batch。
- 发送给 learner 时先把 rollout tensors 放在 CPU，避免 GPU 显存互相挤占。

更高级实现：

- `torch.distributed` process group。
- `broadcast_object_list` 或显式 state dict broadcast。
- 对大 rollout arrays 使用 shared memory。

### 预期效果

主要成本在 rollout/probe。如果先实现 no-render probe，多卡 worker 对剩余真实渲染和 rollout 的加速会更明显。

16 卡预估提速：

- 只做同步 workers，probe 仍完整渲染：`6x-10x`
- no-render probe + 同步 workers：`10x-20x+`

实际提速取决于：

- 每个 scene 的 3DGS 渲染成本
- CPU 预处理开销
- worker 同步等待和长尾
- rollout batch size
- 模型广播开销
- 每张 GPU 上的 env 数量

## 阶段 4：可选的流水线 Actor-Learner

这一阶段应作为可选项，因为它比同步 rollout 更容易改变算法行为。

### 设计

actor 在 learner 更新时继续采样。learner 接收带有 `policy_version` 标记的 rollout batch。

安全控制：

- 丢弃超过最大 staleness 阈值的旧 batch。
- 记录 actor policy version 和 learner policy version。
- 收紧 KL 检查。
- 保持较小的 `update_epochs`。
- 可以只流水线化环境 reset/preload，而不流水线化真正的 policy rollout。

### 风险

这可以提高硬件利用率，但旧策略数据可能降低 PPO 稳定性。如果目标是尽量保持原始效果，它不应该是第一选择。

建议使用条件：

- 同步多卡方案稳定后再考虑。
- 必须用 SR、collision、reward、KL、entropy、return 等指标和单卡 baseline 对比。

## 阶段 5：PPO Batch 和稳定性设置

如果使用正确，这些改动不会破坏 PPO 的 on-policy 性质。

分布式 rollout 后的推荐设置：

- 保持 `update_epochs=1` 或 `2`。
- 根据 rollout batch size，把 `minibatch_size` 从 `4` 提高到 `16`、`32` 或 `64`。
- 保持 KL early stop 开启。
- 保留 rollout policy 生成的 old logprobs，策略更新后不要重新计算 old logprobs。
- 不要跨多个 PPO update 重复使用同一批 rollout 数据。

为什么这仍然是 on-policy：

- `rollout_batch = num_workers * local_num_envs * num_steps` 是当前冻结策略采集的。
- `minibatch_size` 只决定这批新 rollout 数据在 PPO update 时怎么切分。
- 更新结束后，这批数据会被丢弃。

## 阶段 6：性能监控和计时

增加计时器，用于定位剩余瓶颈：

- rollout model forward 时间
- trajectory mode sampling 时间
- probe reward 时间
- 真实 env step/render 时间
- no-render probe step 时间
- GAE/return 计算时间
- PPO update 时间
- worker 等待时间
- 模型广播时间

每个 update 记录：

```text
timing/rollout_s
timing/probe_s
timing/env_step_s
timing/ppo_update_s
timing/broadcast_s
throughput/env_steps_per_s
throughput/probe_steps_per_s
throughput/updates_per_hour
```

这一步很重要，用来确认优化真的打到了瓶颈，而不是只是把开销从一个地方挪到了另一个地方。

## 实施顺序

1. 在 `ReconSimulator` 中添加 `step_no_render()`。
2. 在 `ReconNusPPOEnv` 中添加 `step_reward_only()`。
3. 修改 `TrajectoryProbeReward`，probe 阶段优先使用 `step_reward_only()`。
4. 增加 full-step reward 和 reward-only step 的等价性验证。
5. 检查或修复 episode termination，必要时改为 `now_frame >= final_frame`。
6. 添加 PPO 计时和吞吐日志。
7. 添加 `device_ids` env 设备分配支持。
8. 先实现 2 卡同步 rollout worker 原型。
9. 扩展到 4 卡、8 卡，再到 16 卡。
10. rollout batch size 增大后，再提高 `minibatch_size`。
11. 与当前 baseline 对比训练指标。
12. 同步 rollout 稳定后，再考虑可选的流水线 actor-learner。

## 验收标准

正确性标准：

- no-render probe 对采样 action 返回的 reward 和关键信息与 full probe 一致。
- 真实 rollout observation 仍然由 3DGS 渲染得到。
- PPO rollout batch 带有 policy version 标记。
- learner 只使用当前同步策略版本的数据更新。
- old logprobs 在 update 前生成并保留。
- 每次 PPO update 后丢弃 rollout 数据。

训练指标相对 baseline 不应有明显退化：

- success rate
- collision rate
- deviation rate
- average reward
- average return
- KL
- entropy
- value loss
- probe reward distribution

性能标准：

- 阶段 1 应显著降低 probe 时间。
- 阶段 3 应体现明确的多卡吞吐提升。
- 16 卡下，现实目标至少是 `6x` 提速。
- no-render probe + 同步 workers 后，挑战目标是 `10x-20x+`。

## 关键风险

1. No-render probe reward 不一致。

   应对：增加等价性测试，并保留 full-step fallback 开关。

2. 多进程 CUDA 初始化问题。

   应对：使用 `spawn`，在 worker 进程内部初始化 env，避免 CUDA 初始化后再 fork。

3. 模型广播开销过高。

   应对：每个 PPO update 广播一次，不要每个 step 广播。

4. worker 长尾拖慢同步。

   应对：均衡分配 scene/env，并记录每个 worker 的 rollout 时间。

5. 引入流水线后出现旧策略数据。

   应对：默认保持同步 rollout；只有加了 policy-version 检查后才开启流水线。

## 推荐的第一阶段交付

优先实现阶段 1 和阶段 6：

- `step_no_render()`
- `step_reward_only()`
- probe 使用 reward-only stepping
- full-step fallback 开关
- timing logs
- 等价性检查

这是风险最低、收益最高的改动，并且保留了原项目的 probe 训练思想。

