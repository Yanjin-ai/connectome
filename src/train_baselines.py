import os
import sys

sys.path.insert(0, ".")

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.mlp_baseline import MLPController
from src.random_graph_gnn import RandomGraphController


SEED = 42
TIMESTEPS = 200_000
HIDDEN_DIM = 32

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def make_extractor_class(ControllerClass):
    """Factory: creates a SB3 FeaturesExtractor wrapping ControllerClass."""

    class Extractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, hidden_dim=32):
            super().__init__(observation_space, features_dim=hidden_dim)
            obs_dim = int(np.prod(observation_space.shape))
            self.ctrl = ControllerClass(
                obs_dim=obs_dim,
                act_dim=hidden_dim,
                hidden_dim=hidden_dim,
            )

        def forward(self, obs):
            return self.ctrl(obs)

    return Extractor


configs = [
    ("random", RandomGraphController, "logs/random_eval", "models/random_best"),
    ("mlp", MLPController, "logs/mlp_eval", "models/mlp_best"),
]


for name, Ctrl, log_path, save_path in configs:
    print(f"\n{'=' * 50}")
    print(f"Training {name.upper()} baseline for {TIMESTEPS:,} steps...")

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    policy_kwargs = dict(
        features_extractor_class=make_extractor_class(Ctrl),
        features_extractor_kwargs=dict(hidden_dim=HIDDEN_DIM),
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
        verbose=1,
        seed=SEED,
    )

    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=20,
        eval_freq=5_000,
        log_path=log_path,
        best_model_save_path=save_path,
        deterministic=True,
        verbose=0,
    )

    model.learn(total_timesteps=TIMESTEPS, callback=eval_cb, progress_bar=True)
    model.save(f"models/{name}_ppo_final")

    evals = np.load(f"{log_path}/evaluations.npz")
    timesteps = evals["timesteps"]
    results = evals["results"]
    mean_r = results.mean(axis=1)
    std_r = results.std(axis=1)
    np.save(
        f"results/{name}_curve.npy",
        np.stack([timesteps, mean_r, std_r], axis=0),
    )

    solved = np.where(mean_r >= 450)[0]
    print(f"Final reward : {mean_r[-1]:.1f} ± {std_r[-1]:.1f}")
    print(
        f"First solved : {timesteps[solved[0]]:,} steps"
        if len(solved)
        else "Not solved within budget"
    )
    print(f"Saved: results/{name}_curve.npy")

print("\n=== ALL BASELINES COMPLETE ===")
