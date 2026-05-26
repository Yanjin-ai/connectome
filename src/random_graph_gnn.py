import torch
import torch.nn as nn
import pandas as pd


class RandomGraphController(nn.Module):
    """
    Random-graph GNN policy network.

    Matches ConnectomeController's node counts, trainable layers, and forward
    pass. Only the fixed adjacency matrix is replaced with a reproducible
    random directed graph.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 32,
        n_mp_steps: int = 2,
        adj_path: str = "data/toy_connectome_edges.csv",
        nodes_path: str = "data/toy_connectome_nodes.csv",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_mp_steps = n_mp_steps

        edges_df = pd.read_csv(adj_path)
        self.n_nodes = 30
        self.n_sensor = 6
        self.n_motor = 6
        n_edges = len(edges_df)

        sensor_idx = list(range(0, 6))
        motor_idx = list(range(24, 30))

        self.register_buffer("sensor_idx", torch.tensor(sensor_idx, dtype=torch.long))
        self.register_buffer("motor_idx", torch.tensor(motor_idx, dtype=torch.long))

        # Fixed random adjacency matrix (column-normalised).
        # Preserve the caller's RNG stream so this baseline differs only in A.
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(0)
        possible_edges = torch.tensor(
            [
                (i, j)
                for i in range(self.n_nodes)
                for j in range(self.n_nodes)
                if i != j
            ],
            dtype=torch.long,
        )
        selected = possible_edges[torch.randperm(len(possible_edges))[:n_edges]]
        torch.random.set_rng_state(rng_state)

        A = torch.zeros(self.n_nodes, self.n_nodes)
        A[selected[:, 0], selected[:, 1]] = 1.0
        col_sum = A.sum(0).clamp(min=1e-8)
        A = A / col_sum.unsqueeze(0)
        self.register_buffer("A", A)

        # Learnable layers: identical to ConnectomeController.
        self.sensor_proj = nn.Linear(obs_dim, hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )
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

        H = torch.zeros(
            B,
            self.n_nodes,
            self.hidden_dim,
            device=obs.device,
            dtype=obs.dtype,
        )

        sensor_feat = self.sensor_proj(obs)
        H[:, self.sensor_idx, :] = sensor_feat.unsqueeze(1).expand(
            B, self.n_sensor, self.hidden_dim
        )

        for _ in range(self.n_mp_steps):
            msgs = torch.einsum("ij,bjh->bih", self.A, H)
            H = self.update_mlp(torch.cat([H, msgs], dim=-1))

        motor_states = H[:, self.motor_idx, :]
        agg = motor_states.mean(dim=1)

        return self.output_head(agg)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
