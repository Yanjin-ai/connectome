import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.connectome_gnn import ConnectomeController


os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs/connectome_eval", exist_ok=True)


class ConnectomeFeaturesExtractor(BaseFeaturesExtractor):
    """Wraps ConnectomeController as SB3 features extractor."""

    def __init__(self, observation_space, hidden_dim: int = 32):
        super().__init__(observation_space, features_dim=hidden_dim)
        obs_dim = int(np.prod(observation_space.shape))
        self.gnn = ConnectomeController(
            obs_dim=obs_dim,
            act_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_mp_steps=2,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.gnn(obs)


class ConnectomeMlpExtractor(nn.Module):
    """Minimal SB3-compatible extractor used only if the feature wrapper fails."""

    def __init__(self, obs_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.gnn = ConnectomeController(
            obs_dim=obs_dim,
            act_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_mp_steps=2,
        )
        self.latent_dim_pi = hidden_dim
        self.latent_dim_vf = hidden_dim

    def forward(self, features: torch.Tensor):
        latent = self.gnn(features)
        return latent, latent

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.gnn(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.gnn(features)


class ConnectomePolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            activation_fn=nn.Tanh,
            **kwargs,
        )

    def _build_mlp_extractor(self):
        self.mlp_extractor = ConnectomeMlpExtractor(
            obs_dim=self.observation_space.shape[0],
            hidden_dim=HIDDEN_DIM,
        )


SEED = 42
TIMESTEPS = 200_000
HIDDEN_DIM = 32

env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")

policy_kwargs = dict(
    features_extractor_class=ConnectomeFeaturesExtractor,
    features_extractor_kwargs=dict(hidden_dim=HIDDEN_DIM),
    net_arch=[],
    activation_fn=nn.Tanh,
)

try:
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
except Exception:
    print("FALLBACK USED")
    model = PPO(
        ConnectomePolicy,
        env,
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

evals = np.load("logs/connectome_eval/evaluations.npz")
timesteps = evals["timesteps"]
results = evals["results"]
mean_r = results.mean(axis=1)
std_r = results.std(axis=1)

np.save(
    "results/connectome_curve.npy",
    np.stack([timesteps, mean_r, std_r], axis=0),
)

print("\n=== CONNECTOME TRAINING COMPLETE ===")
print(f"Final eval reward: {mean_r[-1]:.1f} ± {std_r[-1]:.1f}")
print(f"Max eval reward:   {mean_r.max():.1f}")
solved = np.where(mean_r >= 450)[0]
print(
    f"First solved at:   {timesteps[solved[0]]:,} steps"
    if len(solved)
    else "Not solved"
)
print("Saved: results/connectome_curve.npy")
