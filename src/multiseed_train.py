import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.connectome_gnn import ConnectomeController
from src.mlp_baseline import MLPController
from src.random_graph_gnn import RandomGraphController


SEEDS = [42, 123, 456, 789, 1024]
TIMESTEPS = 200_000
HIDDEN = 32
ARCHS = {
    "connectome": ConnectomeController,
    "random": RandomGraphController,
    "mlp": MLPController,
}


def make_extractor(Ctrl, hidden):
    class E(BaseFeaturesExtractor):
        def __init__(self, obs_space, hidden_dim=hidden):
            super().__init__(obs_space, features_dim=hidden_dim)
            self.ctrl = Ctrl(
                obs_dim=int(np.prod(obs_space.shape)),
                act_dim=hidden_dim,
                hidden_dim=hidden_dim,
            )

        def forward(self, x):
            return self.ctrl(x)

    return E


os.makedirs("results", exist_ok=True)

for arch, Ctrl in ARCHS.items():
    for seed in SEEDS:
        out_path = Path(f"results/{arch}_seed{seed}.npy")
        if out_path.exists():
            print(f"[SKIP] {arch} seed={seed} already done")
            continue

        print(f"\n>>> Training {arch.upper()} seed={seed}")
        log_dir = f"logs/{arch}_seed{seed}"
        save_dir = f"models/{arch}_seed{seed}_best"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        env = gym.make("CartPole-v1")
        eval_env = gym.make("CartPole-v1")

        policy_kwargs = dict(
            features_extractor_class=make_extractor(Ctrl, HIDDEN),
            features_extractor_kwargs=dict(hidden_dim=HIDDEN),
            net_arch=[],
            activation_fn=torch.nn.Tanh,
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
            verbose=0,
            seed=seed,
        )

        eval_cb = EvalCallback(
            eval_env,
            n_eval_episodes=20,
            eval_freq=5_000,
            log_path=log_dir,
            best_model_save_path=save_dir,
            deterministic=True,
            verbose=0,
        )

        model.learn(total_timesteps=TIMESTEPS, callback=eval_cb, progress_bar=True)

        evals = np.load(f"{log_dir}/evaluations.npz")
        t = evals["timesteps"]
        r = evals["results"].mean(axis=1)
        s = evals["results"].std(axis=1)
        np.save(out_path, np.stack([t, r, s]))
        print(f"    Saved {out_path}  final_reward={r[-1]:.1f}")

print("\n=== ALL MULTI-SEED RUNS COMPLETE ===")
