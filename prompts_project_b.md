# Project B — Codex Execution Prompts
# Connectome-shaped GNN Controller vs Baselines

4 个 prompt 按顺序执行。
每个 prompt 是独立的，出错可以单独重跑。

卡点说明：
- B1 → B2 依赖：B1 必须产出 src/connectome_gnn.py 且 forward pass 验证通过
- B2 → B3 依赖：B2 必须产出 results/connectome_curve.npy（training logs）
- B3 → B4 依赖：B3 必须产出 random_curve.npy 和 mlp_curve.npy
- B4 是最终图，所有 npy 都存在才能跑

---

## Prompt B1 — GNN 架构实现 + 验证

> **目的**：写出 ConnectomeGNN 类，用 toy connectome 拓扑做计算图，验证 forward pass 无误。
> **无外部依赖**：只用 torch + pandas（不需要 PyTorch Geometric，避免安装问题）。
> **预计时间**：5 分钟。
> **卡点**：sensor 节点 index 赋值；forward pass shape 检查。

```
You are implementing Project B of a computational neuroscience + embodied AI research project.

Working directory: /Volumes/SANDISK ELE/embodied summer program
Required packages: torch, pandas, numpy  (all installed)
Data files:
  data/toy_connectome_nodes.csv   (columns: node, layer)   30 nodes
  data/toy_connectome_edges.csv   (columns: src, dst, weight)

Context: Project A showed that the fruit fly connectome has FF=10.9%, FB=6.4%, Lateral=82.8%.
The GNN controller must preserve this structure. We use a 30-node toy graph (sensor/inter/motor)
as a small-scale proxy for the real connectome, exactly as in FlyGM (arXiv 2602.17997).

Architecture spec (from Project A controller_prior_report.txt):
- Message-passing depth: 2 hops
- Hub nodes: wider embedding (handled via shared update MLP)
- Feedback edges: keep in adjacency (they are in toy CSV as motor→inter edges)
- Lateral edges: keep in adjacency (inter→inter recurrence in toy CSV)
- Node types: sensor receives obs, motor produces action, inter does message passing

─────────────────────────────────────────────
TASK: Create src/connectome_gnn.py
─────────────────────────────────────────────
mkdir -p src results models logs

The file must implement this class exactly:

```python
# src/connectome_gnn.py
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

DATA = Path("data")

class ConnectomeController(nn.Module):
    """
    Connectome-shaped GNN policy network.
    
    Mirrors FlyGM (arXiv 2602.17997) at small scale:
    - Sensor nodes receive observation (afferent)
    - Inter nodes do message passing (intrinsic)
    - Motor nodes produce action (efferent)
    
    Pure PyTorch, no PyTorch Geometric required.
    Fixed adjacency from toy_connectome_edges.csv.
    """
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dim: int = 32, n_mp_steps: int = 2,
                 adj_path: str = "data/toy_connectome_edges.csv",
                 nodes_path: str = "data/toy_connectome_nodes.csv"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_mp_steps = n_mp_steps

        # ── Load graph ──────────────────────────────────────────────────
        nodes_df = pd.read_csv(nodes_path)
        edges_df = pd.read_csv(adj_path)
        self.n_nodes = len(nodes_df)

        node_order = nodes_df["node"].tolist()
        node_idx = {n: i for i, n in enumerate(node_order)}
        layers = nodes_df["layer"].values

        sensor_idx = [i for i, l in enumerate(layers) if l == "sensor"]
        motor_idx  = [i for i, l in enumerate(layers) if l == "motor"]
        self.n_sensor = len(sensor_idx)
        self.n_motor  = len(motor_idx)

        self.register_buffer("sensor_idx", torch.tensor(sensor_idx, dtype=torch.long))
        self.register_buffer("motor_idx",  torch.tensor(motor_idx,  dtype=torch.long))

        # Fixed weighted adjacency matrix (column-normalised)
        A = torch.zeros(self.n_nodes, self.n_nodes)
        for _, row in edges_df.iterrows():
            i, j = node_idx[row["src"]], node_idx[row["dst"]]
            A[i, j] = float(row["weight"])
        col_sum = A.sum(0).clamp(min=1e-8)
        A = A / col_sum.unsqueeze(0)   # column normalise
        self.register_buffer("A", A)   # (n_nodes, n_nodes)

        # ── Learnable layers ─────────────────────────────────────────────
        # Sensor input projection: each sensor node gets the full obs
        self.sensor_proj = nn.Linear(obs_dim, hidden_dim)

        # Shared message-passing update MLP
        # Input: concat(h_v, aggregated_msgs_v) = 2 * hidden_dim
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )

        # Output head: aggregate motor states → action logits
        self.output_head = nn.Linear(hidden_dim, act_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs : (B, obs_dim)
        returns : (B, act_dim)
        """
        B = obs.shape[0]

        # Initialise all node states to zero
        H = torch.zeros(B, self.n_nodes, self.hidden_dim,
                        device=obs.device, dtype=obs.dtype)

        # Project obs into sensor nodes (all sensors share the same projection)
        sensor_feat = self.sensor_proj(obs)               # (B, hidden_dim)
        H[:, self.sensor_idx, :] = sensor_feat.unsqueeze(1).expand(
            B, self.n_sensor, self.hidden_dim)

        # Message passing rounds
        for _ in range(self.n_mp_steps):
            # A: (n_nodes, n_nodes)   H: (B, n_nodes, hidden_dim)
            # msgs[b, i, :] = sum_j A[i,j] * H[b,j,:]
            msgs = torch.einsum("ij,bjh->bih", self.A, H)   # (B, n_nodes, hidden_dim)
            H = self.update_mlp(torch.cat([H, msgs], dim=-1))

        # Aggregate motor node states (mean pooling)
        motor_states = H[:, self.motor_idx, :]    # (B, n_motor, hidden_dim)
        agg = motor_states.mean(dim=1)            # (B, hidden_dim)

        return self.output_head(agg)              # (B, act_dim)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

─────────────────────────────────────────────
VERIFICATION (run this immediately after saving):
─────────────────────────────────────────────

```python
import sys, torch
sys.path.insert(0, ".")
from src.connectome_gnn import ConnectomeController

# CartPole: obs_dim=4, act_dim=2
model = ConnectomeController(obs_dim=4, act_dim=2, hidden_dim=32, n_mp_steps=2)

dummy_obs = torch.randn(8, 4)
out = model(dummy_obs)
assert out.shape == (8, 2), f"Bad shape: {out.shape}"

print(f"✓ Forward pass OK: input {dummy_obs.shape} → output {out.shape}")
print(f"✓ Trainable parameters: {model.count_params():,}")
print(f"  n_nodes={model.n_nodes}, n_sensor={model.n_sensor}, n_motor={model.n_motor}")
print(f"  Adjacency matrix shape: {model.A.shape}")
print(f"  Non-zero edges in A: {(model.A > 0).sum().item()}")

# Test gradient flow
loss = out.sum()
loss.backward()
print("✓ Backward pass OK")

# Test batches of different sizes
for bs in [1, 16, 64]:
    o = model(torch.randn(bs, 4))
    assert o.shape == (bs, 2)
print("✓ Variable batch size OK")
```

EXPECTED OUTPUT:
  ✓ Forward pass OK: input torch.Size([8, 4]) → output torch.Size([8, 2])
  ✓ Trainable parameters: ~2300
  ✓ Backward pass OK
  ✓ Variable batch size OK

Save the file as src/connectome_gnn.py. If any assertion fails, fix and re-verify.
Print the complete contents of src/connectome_gnn.py when done.
```

---

## Prompt B2 — PPO 训练：Connectome 控制器

> **目的**：用 stable-baselines3 的 PPO 训练 ConnectomeController，保存 training log。
> **前置条件**：src/connectome_gnn.py 存在且验证通过。
> **预计时间**：10–20 分钟（200k steps, CartPole）。
> **卡点**：SB3 自定义 policy 的接口写法。如果 FeaturesExtractor 方式不工作，使用备用方案（见文末）。

```
You are training Project B — the connectome-shaped GNN controller on CartPole-v1.

Working directory: /Volumes/SANDISK ELE/embodied summer program
Required packages: stable-baselines3[extra] gymnasium torch
  Install if missing: pip install "stable-baselines3[extra]" gymnasium

File: src/connectome_gnn.py  (ConnectomeController class, already written)

─────────────────────────────────────────────
TASK: Create src/train_connectome.py and run it.
─────────────────────────────────────────────

```python
# src/train_connectome.py
import sys, os
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from pathlib import Path

from src.connectome_gnn import ConnectomeController

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs/connectome_eval", exist_ok=True)

# ── SB3 FeaturesExtractor wrapper ─────────────────────────────────────
class ConnectomeFeaturesExtractor(BaseFeaturesExtractor):
    """Wraps ConnectomeController as SB3 features extractor."""
    def __init__(self, observation_space, hidden_dim: int = 32):
        super().__init__(observation_space, features_dim=hidden_dim)
        obs_dim = int(np.prod(observation_space.shape))
        self.gnn = ConnectomeController(
            obs_dim=obs_dim,
            act_dim=hidden_dim,   # output = motor aggregate (hidden_dim vector)
            hidden_dim=hidden_dim,
            n_mp_steps=2,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.gnn(obs)   # (B, hidden_dim)

# ── Training ───────────────────────────────────────────────────────────
SEED       = 42
TIMESTEPS  = 200_000
HIDDEN_DIM = 32

env      = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")

policy_kwargs = dict(
    features_extractor_class=ConnectomeFeaturesExtractor,
    features_extractor_kwargs=dict(hidden_dim=HIDDEN_DIM),
    net_arch=[],          # NO additional MLP on top — GNN IS the policy backbone
    activation_fn=nn.Tanh,
)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
    seed=SEED,
    tensorboard_log="logs/tb/",
)

eval_cb = EvalCallback(
    eval_env,
    n_eval_episodes=20,
    eval_freq=5_000,
    log_path="logs/connectome_eval",
    best_model_save_path="models/connectome_best",
    deterministic=True,
    verbose=0,
)

print(f"Training ConnectomeController for {TIMESTEPS:,} steps...")
model.learn(total_timesteps=TIMESTEPS, callback=eval_cb, progress_bar=True)
model.save("models/connectome_ppo_final")

# ── Parse and save curve ────────────────────────────────────────────────
evals = np.load("logs/connectome_eval/evaluations.npz")
timesteps = evals["timesteps"]
results   = evals["results"]          # shape (n_checkpoints, n_eval_episodes)
mean_r    = results.mean(axis=1)
std_r     = results.std(axis=1)

np.save("results/connectome_curve.npy",
        np.stack([timesteps, mean_r, std_r], axis=0))

print("\n=== CONNECTOME TRAINING COMPLETE ===")
print(f"Final eval reward: {mean_r[-1]:.1f} ± {std_r[-1]:.1f}")
print(f"Max eval reward:   {mean_r.max():.1f}")
# Steps to first solve (reward >= 450)
solved = np.where(mean_r >= 450)[0]
print(f"First solved at:   {timesteps[solved[0]]:,} steps" if len(solved) else "Not solved")
print(f"Saved: results/connectome_curve.npy")
```

Run: python3 src/train_connectome.py

─────────────────────────────────────────────
FALLBACK if FeaturesExtractor approach fails:
─────────────────────────────────────────────
If SB3 raises errors about policy_kwargs or features_extractor, use this
minimal custom policy instead:

```python
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution

class ConnectomePolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule,
                         net_arch=[], **kwargs)
    
    def _build_mlp_extractor(self):
        self.mlp_extractor = ConnectomeController(
            obs_dim=self.observation_space.shape[0],
            act_dim=32, hidden_dim=32
        )
        self.mlp_extractor.latent_dim_pi = 32
        self.mlp_extractor.latent_dim_vf = 32
```

Print "FALLBACK USED" and continue if this path is taken.

EXPECTED OUTPUT:
  Training output showing reward increasing over 200k steps.
  Final: results/connectome_curve.npy  (shape: 3 × n_checkpoints)
  models/connectome_ppo_final.zip
```

---

## Prompt B3 — 两个 Baseline 实现 + 训练

> **目的**：实现随机图 GNN 和 MLP 两个对比组，用完全相同的训练配置训练。
> **前置条件**：Prompt B2 完成，src/connectome_gnn.py 存在。
> **预计时间**：20–40 分钟（两个模型各 200k steps）。
> **卡点**：参数量要和 ConnectomeController 对齐；随机图构造要可复现。

```
You are implementing two baselines for Project B, to compare against the connectome GNN.

Working directory: /Volumes/SANDISK ELE/embodied summer program
File: src/connectome_gnn.py  (reference — DO NOT modify)

Context from Project A:
  ConnectomeController has ~2300 trainable parameters.
  The toy connectome has 30 nodes, with specific sensor/inter/motor wiring.
  Both baselines must use identical training hyperparameters to the connectome model.

─────────────────────────────────────────────
BASELINE 1: Random Graph GNN
─────────────────────────────────────────────
Create src/random_graph_gnn.py

SAME class interface as ConnectomeController, SAME forward pass logic.
The ONLY difference: the adjacency matrix A is randomly wired instead of
coming from the real toy connectome.

Construction:
  n_nodes = 30  (same as connectome)
  n_sensor = 6, n_inter = 18, n_motor = 6  (same type counts)
  n_edges = same as toy connectome (count edges in toy_connectome_edges.csv)
  
  Wiring:
    torch.manual_seed(0)  # reproducible random graph
    # Sample (n_edges) random directed edges uniformly from all possible (i,j) pairs
    # excluding self-loops
    # Apply same column normalization as ConnectomeController
    
  The sensor/motor/inter NODE INDICES stay the same (0-5 sensor, 6-23 inter, 24-29 motor).
  Only the EDGES change.

Verify:
  model = RandomGraphController(obs_dim=4, act_dim=2)
  out = model(torch.randn(8, 4))
  assert out.shape == (8, 2)
  print(f"Random graph params: {model.count_params():,}")
  # Should be identical to ConnectomeController param count

─────────────────────────────────────────────
BASELINE 2: MLP
─────────────────────────────────────────────
Create src/mlp_baseline.py

Standard 2-layer MLP with matched parameter count.

```python
class MLPController(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 32):
        super().__init__()
        # Count target params from ConnectomeController
        from src.connectome_gnn import ConnectomeController
        ref = ConnectomeController(obs_dim=obs_dim, act_dim=act_dim)
        target_params = ref.count_params()
        
        # Solve for h: obs_dim*h + h + h*h + h + h*act_dim + act_dim ≈ target
        # Approximate: h ≈ sqrt(target / 2)
        import math
        h = max(32, int(math.sqrt(target_params / 2)))
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, act_dim),
        )
        actual = sum(p.numel() for p in self.parameters())
        print(f"MLP hidden={h}, params={actual:,} (target={target_params:,})")
    
    def forward(self, obs):
        return self.net(obs)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
```

─────────────────────────────────────────────
TRAINING: Create src/train_baselines.py
─────────────────────────────────────────────

```python
# src/train_baselines.py
# Trains both baselines with identical config as train_connectome.py

import sys, os
sys.path.insert(0, ".")
import numpy as np, torch, gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from pathlib import Path

from src.random_graph_gnn import RandomGraphController
from src.mlp_baseline import MLPController

SEED = 42; TIMESTEPS = 200_000; HIDDEN_DIM = 32

def make_extractor_class(ControllerClass):
    """Factory: creates a SB3 FeaturesExtractor wrapping ControllerClass."""
    class Extractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, hidden_dim=32):
            super().__init__(observation_space, features_dim=hidden_dim)
            obs_dim = int(np.prod(observation_space.shape))
            self.ctrl = ControllerClass(obs_dim=obs_dim, act_dim=hidden_dim,
                                        hidden_dim=hidden_dim)
        def forward(self, obs):
            return self.ctrl(obs)
    return Extractor

configs = [
    ("random",  RandomGraphController, "logs/random_eval",  "models/random_best"),
    ("mlp",     MLPController,         "logs/mlp_eval",     "models/mlp_best"),
]

for name, Ctrl, log_path, save_path in configs:
    print(f"\n{'='*50}")
    print(f"Training {name.upper()} baseline for {TIMESTEPS:,} steps...")
    
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    env      = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    
    policy_kwargs = dict(
        features_extractor_class=make_extractor_class(Ctrl),
        features_extractor_kwargs=dict(hidden_dim=HIDDEN_DIM),
        net_arch=[], activation_fn=torch.nn.Tanh,
    )
    model = PPO("MlpPolicy", env,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0,
                verbose=1, seed=SEED)
    
    eval_cb = EvalCallback(eval_env, n_eval_episodes=20, eval_freq=5_000,
                           log_path=log_path, best_model_save_path=save_path,
                           deterministic=True, verbose=0)
    
    model.learn(total_timesteps=TIMESTEPS, callback=eval_cb, progress_bar=True)
    model.save(f"models/{name}_ppo_final")
    
    evals     = np.load(f"{log_path}/evaluations.npz")
    timesteps = evals["timesteps"]
    results   = evals["results"]
    mean_r    = results.mean(axis=1)
    std_r     = results.std(axis=1)
    np.save(f"results/{name}_curve.npy",
            np.stack([timesteps, mean_r, std_r], axis=0))
    
    solved = np.where(mean_r >= 450)[0]
    print(f"Final reward : {mean_r[-1]:.1f} ± {std_r[-1]:.1f}")
    print(f"First solved : {timesteps[solved[0]]:,} steps" if len(solved) else "Not solved within budget")
    print(f"Saved: results/{name}_curve.npy")

print("\n=== ALL BASELINES COMPLETE ===")
```

Run: python3 src/train_baselines.py

EXPECTED OUTPUT:
  results/random_curve.npy
  results/mlp_curve.npy
  models/random_ppo_final.zip
  models/mlp_ppo_final.zip
  Console: final reward and solve step for each baseline
```

---

## Prompt B4 — Multi-seed 实验 + 最终对比图

> **目的**：5 个随机种子 × 3 个架构跑完整实验，生成论文级对比图。
> **前置条件**：B1–B3 完成，results/ 下有 3 个 curve.npy 文件。
> **预计时间**：1–2 小时（15 次训练），之后生成图 2 分钟。
> **卡点**：训练时间长——脚本设计为可断点续跑（跳过已存在的结果文件）。

```
You are running the final multi-seed experiment and generating the Project B comparison figure.

Working directory: /Volumes/SANDISK ELE/embodied summer program
Files needed: src/connectome_gnn.py, src/random_graph_gnn.py, src/mlp_baseline.py

─────────────────────────────────────────────
STEP 1: Create src/multiseed_train.py
─────────────────────────────────────────────

```python
# src/multiseed_train.py
# Trains all 3 architectures × 5 seeds. Skips already-completed runs.

import sys, os
sys.path.insert(0, ".")
import numpy as np, torch, gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from pathlib import Path

from src.connectome_gnn    import ConnectomeController
from src.random_graph_gnn  import RandomGraphController
from src.mlp_baseline      import MLPController

SEEDS     = [42, 123, 456, 789, 1024]
TIMESTEPS = 200_000
HIDDEN    = 32
ARCHS     = {
    "connectome": ConnectomeController,
    "random":     RandomGraphController,
    "mlp":        MLPController,
}

def make_extractor(Ctrl, hidden):
    class E(BaseFeaturesExtractor):
        def __init__(self, obs_space, hidden_dim=hidden):
            super().__init__(obs_space, features_dim=hidden_dim)
            self.ctrl = Ctrl(obs_dim=int(np.prod(obs_space.shape)),
                             act_dim=hidden_dim, hidden_dim=hidden_dim)
        def forward(self, x):
            return self.ctrl(x)
    return E

for arch, Ctrl in ARCHS.items():
    for seed in SEEDS:
        out_path = Path(f"results/{arch}_seed{seed}.npy")
        if out_path.exists():
            print(f"[SKIP] {arch} seed={seed} already done")
            continue
        
        print(f"\n>>> Training {arch.upper()} seed={seed}")
        log_dir  = f"logs/{arch}_seed{seed}"
        save_dir = f"models/{arch}_seed{seed}_best"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        
        env = gym.make("CartPole-v1")
        eval_env = gym.make("CartPole-v1")
        
        policy_kwargs = dict(
            features_extractor_class=make_extractor(Ctrl, HIDDEN),
            features_extractor_kwargs=dict(hidden_dim=HIDDEN),
            net_arch=[], activation_fn=torch.nn.Tanh,
        )
        model = PPO("MlpPolicy", env,
                    policy_kwargs=policy_kwargs,
                    learning_rate=3e-4, n_steps=2048, batch_size=64,
                    n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.0,
                    verbose=0, seed=seed)
        
        eval_cb = EvalCallback(eval_env, n_eval_episodes=20, eval_freq=5_000,
                               log_path=log_dir,
                               best_model_save_path=save_dir,
                               deterministic=True, verbose=0)
        
        model.learn(total_timesteps=TIMESTEPS, callback=eval_cb, progress_bar=True)
        
        evals = np.load(f"{log_dir}/evaluations.npz")
        t     = evals["timesteps"]
        r     = evals["results"].mean(axis=1)
        s     = evals["results"].std(axis=1)
        np.save(out_path, np.stack([t, r, s]))
        print(f"    Saved {out_path}  final_reward={r[-1]:.1f}")

print("\n=== ALL MULTI-SEED RUNS COMPLETE ===")
```

Run: python3 src/multiseed_train.py
(Can be interrupted and re-run — already-finished seeds are skipped)

─────────────────────────────────────────────
STEP 2: Create src/plot_results.py and run it
─────────────────────────────────────────────

```python
# src/plot_results.py
import sys
sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SEEDS = [42, 123, 456, 789, 1024]
ARCHS = {
    "connectome": ("Connectome GNN", "#2196F3"),
    "random":     ("Random Graph",   "#FF9800"),
    "mlp":        ("MLP",            "#9E9E9E"),
}
SOLVE_THRESHOLD = 450   # CartPole reward threshold for "solved"

# ── Load all curves ──────────────────────────────────────────────────
data = {}
for arch in ARCHS:
    curves = []
    for seed in SEEDS:
        p = Path(f"results/{arch}_seed{seed}.npy")
        if p.exists():
            curves.append(np.load(p))   # (3, T) = [timesteps, mean_r, std_r]
    if curves:
        # Align to common timesteps (use first seed's timestep array)
        ts = curves[0][0]
        means = np.stack([c[1] for c in curves if len(c[0]) == len(ts)])
        data[arch] = {"ts": ts, "mean": means.mean(0), "std": means.std(0), "all": means}
    else:
        print(f"WARNING: no data for {arch}")

# ── Compute metrics ──────────────────────────────────────────────────
metrics = {}
for arch, d in data.items():
    ts, all_r = d["ts"], d["all"]
    # Sample efficiency: median steps to first reach SOLVE_THRESHOLD
    solve_steps = []
    for r in all_r:
        idx = np.where(r >= SOLVE_THRESHOLD)[0]
        solve_steps.append(ts[idx[0]] if len(idx) else ts[-1])
    # Asymptotic: mean reward in last 30% of training
    last_30 = int(0.7 * len(ts))
    asym_means = all_r[:, last_30:].mean(axis=1)
    stab_stds  = all_r[:, last_30:].std(axis=1)
    metrics[arch] = {
        "solve_steps":  np.array(solve_steps),
        "asym_reward":  asym_means,
        "stability":    stab_stds,
    }

# ── Plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Project B: Connectome GNN vs Baselines (CartPole-v1, 5 seeds)",
             fontsize=13, fontweight="bold")

# A. Learning curves
ax = axes[0, 0]
for arch, (label, color) in ARCHS.items():
    if arch not in data: continue
    d = data[arch]
    ax.plot(d["ts"], d["mean"], color=color, lw=2, label=label)
    ax.fill_between(d["ts"],
                    d["mean"] - d["std"],
                    d["mean"] + d["std"],
                    color=color, alpha=0.15)
ax.axhline(SOLVE_THRESHOLD, color="black", lw=1, ls="--", alpha=0.5, label=f"Solved ({SOLVE_THRESHOLD})")
ax.set_xlabel("Training timesteps"); ax.set_ylabel("Mean eval reward")
ax.set_title("A. Learning curves (mean ± std over 5 seeds)")
ax.legend(fontsize=9); ax.set_ylim(0, 520)

# B. Sample efficiency (steps to solve)
ax = axes[0, 1]
positions = range(len(ARCHS))
for pos, (arch, (label, color)) in zip(positions, ARCHS.items()):
    if arch not in metrics: continue
    s = metrics[arch]["solve_steps"]
    ax.bar(pos, s.mean(), color=color, alpha=0.8, width=0.6)
    ax.errorbar(pos, s.mean(), yerr=s.std(), color="black", capsize=5, lw=2)
ax.set_xticks(list(positions))
ax.set_xticklabels([ARCHS[a][0] for a in ARCHS], fontsize=10)
ax.set_ylabel("Timesteps to reach reward ≥ 450")
ax.set_title("B. Sample efficiency (lower = better)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# C. Asymptotic reward
ax = axes[1, 0]
for pos, (arch, (label, color)) in zip(positions, ARCHS.items()):
    if arch not in metrics: continue
    v = metrics[arch]["asym_reward"]
    ax.bar(pos, v.mean(), color=color, alpha=0.8, width=0.6)
    ax.errorbar(pos, v.mean(), yerr=v.std(), color="black", capsize=5, lw=2)
ax.set_xticks(list(positions))
ax.set_xticklabels([ARCHS[a][0] for a in ARCHS], fontsize=10)
ax.set_ylabel("Mean reward (last 30% of training)")
ax.set_title("C. Asymptotic performance")
ax.set_ylim(0, 520)

# D. Stability (lower std = more stable)
ax = axes[1, 1]
for pos, (arch, (label, color)) in zip(positions, ARCHS.items()):
    if arch not in metrics: continue
    v = metrics[arch]["stability"]
    ax.bar(pos, v.mean(), color=color, alpha=0.8, width=0.6)
    ax.errorbar(pos, v.mean(), yerr=v.std(), color="black", capsize=5, lw=2)
ax.set_xticks(list(positions))
ax.set_xticklabels([ARCHS[a][0] for a in ARCHS], fontsize=10)
ax.set_ylabel("Reward std (last 30% of training)")
ax.set_title("D. Policy stability (lower = better)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("results/fig_project_b_comparison.png", dpi=150)
plt.close(fig)
print("Saved: results/fig_project_b_comparison.png")

# ── Print summary table ───────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Architecture':<18} {'SolveSteps':>12} {'FinalReward':>12} {'Stability':>10}")
print("="*65)
for arch, (label, _) in ARCHS.items():
    if arch not in metrics: continue
    m = metrics[arch]
    ss  = m["solve_steps"]
    ar  = m["asym_reward"]
    st  = m["stability"]
    print(f"{label:<18} {ss.mean()/1000:>8.0f}k±{ss.std()/1000:.0f}k  "
          f"{ar.mean():>8.1f}±{ar.std():.1f}  {st.mean():>8.1f}±{st.std():.1f}")
print("="*65)
```

Run: python3 src/plot_results.py

EXPECTED OUTPUT:
  results/fig_project_b_comparison.png   ← the main result figure
  Console: summary table showing connectome vs random vs MLP across 4 metrics

  If connectome GNN outperforms on any metric (especially sample efficiency or stability),
  that directly supports the core hypothesis of this project.
  Print a one-paragraph interpretation of the results when done.
```

---

## 执行顺序 + 依赖关系

```
B1  (5 min)   → 验证 GNN 架构
    ↓
B2  (20 min)  → 训练 connectome 控制器，得到 results/connectome_curve.npy
    ↓
B3  (40 min)  → 训练两个 baseline，得到 results/random_curve.npy + mlp_curve.npy
    ↓
B4  (1-2 hr)  → 5-seed 全量实验 + 生成最终图
               (B4 Step1 可断点续跑，B4 Step2 最终出图)
```

## 最终产出

运行完成后，你有这些申请材料：

| 文件 | 用途 |
|------|------|
| `results/fig_project_b_comparison.png` | 论文/Portfolio 里的主图 |
| `results/*_curve.npy` | 原始数据，可以随时重新出图 |
| `models/connectome_ppo_final.zip` | 可演示的训练好的模型 |
| `data/real/controller_prior_report.txt` | Project A → B 的设计依据文档 |

## 申请材料里的一句话总结（跑完后填数字）

> "We trained a connectome-topology GNN controller (30 nodes, sensor/inter/motor,
> FF=10.9% / Lateral=82.8%) with PPO on CartPole-v1 and compared it against
> an Erdős–Rényi random graph and an MLP of matched parameter count.
> The connectome controller solved the task in X±Y k steps vs Z±W k for random
> and V±U k for MLP, replicating the sample-efficiency advantage reported by
> FlyGM (arXiv 2602.17997) at small scale."
