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

        # -- Load graph --------------------------------------------------
        nodes_df = pd.read_csv(nodes_path)
        edges_df = pd.read_csv(adj_path)
        self.n_nodes = len(nodes_df)

        node_order = nodes_df["node"].tolist()
        node_idx = {n: i for i, n in enumerate(node_order)}
        layers = nodes_df["layer"].values

        sensor_idx = [i for i, l in enumerate(layers) if l == "sensor"]
        motor_idx = [i for i, l in enumerate(layers) if l == "motor"]
        self.n_sensor = len(sensor_idx)
        self.n_motor = len(motor_idx)

        self.register_buffer("sensor_idx", torch.tensor(sensor_idx, dtype=torch.long))
        self.register_buffer("motor_idx", torch.tensor(motor_idx, dtype=torch.long))

        # Fixed weighted adjacency matrix (column-normalised)
        A = torch.zeros(self.n_nodes, self.n_nodes)
        for _, row in edges_df.iterrows():
            i, j = node_idx[row["src"]], node_idx[row["dst"]]
            A[i, j] = float(row["weight"])
        col_sum = A.sum(0).clamp(min=1e-8)
        A = A / col_sum.unsqueeze(0)  # column normalise
        self.register_buffer("A", A)  # (n_nodes, n_nodes)

        # -- Learnable layers -------------------------------------------
        # Sensor input projection: each sensor node gets the full obs
        self.sensor_proj = nn.Linear(obs_dim, hidden_dim)

        # Shared message-passing update MLP
        # Input: concat(h_v, aggregated_msgs_v) = 2 * hidden_dim
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )

        # Output head: aggregate motor states -> action logits
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
        H = torch.zeros(
            B,
            self.n_nodes,
            self.hidden_dim,
            device=obs.device,
            dtype=obs.dtype,
        )

        # Project obs into sensor nodes (all sensors share the same projection)
        sensor_feat = self.sensor_proj(obs)  # (B, hidden_dim)
        H[:, self.sensor_idx, :] = sensor_feat.unsqueeze(1).expand(
            B, self.n_sensor, self.hidden_dim
        )

        # Message passing rounds
        for _ in range(self.n_mp_steps):
            # A: (n_nodes, n_nodes)   H: (B, n_nodes, hidden_dim)
            # msgs[b, i, :] = sum_j A[i,j] * H[b,j,:]
            msgs = torch.einsum("ij,bjh->bih", self.A, H)  # (B, n_nodes, hidden_dim)
            H = self.update_mlp(torch.cat([H, msgs], dim=-1))

        # Aggregate motor node states (mean pooling)
        motor_states = H[:, self.motor_idx, :]  # (B, n_motor, hidden_dim)
        agg = motor_states.mean(dim=1)  # (B, hidden_dim)

        return self.output_head(agg)  # (B, act_dim)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
