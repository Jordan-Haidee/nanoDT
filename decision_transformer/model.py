from datetime import datetime
from pathlib import Path
from typing import Any
import torch
import gymnasium as gym
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .trajectory import DTDataset, load_trajectories


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_dim: int,
        context_length: int,
        num_gpt_layers: int,
        num_head: int,
        max_timesteps: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_length = context_length

        # build input embedding layers
        self.state_embedding_layer = nn.Linear(state_dim, embedding_dim)
        self.action_embedding_layer = nn.Linear(action_dim, embedding_dim)
        self.return_embedding_layer = nn.Linear(1, embedding_dim)
        self.timestep_embedding_layer = nn.Embedding(max_timesteps, embedding_dim)

        # build laayer-norm layer
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # build gpt layers
        gpt_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_head,
            dim_feedforward=embedding_dim * 4,
            batch_first=True,
        )
        self.gpt_model = nn.TransformerEncoder(gpt_layer, num_layers=num_gpt_layers)

        # build action predictor
        self.action_predictor = nn.Sequential(
            nn.Linear(embedding_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, r, s, a, t):
        batch_size, seq_length, _ = s.shape

        # input embedding
        timestep_embedding = self.timestep_embedding_layer(t)
        return_embedding = self.return_embedding_layer(r) + timestep_embedding
        state_embedding = self.state_embedding_layer(s) + timestep_embedding
        action_embedding = self.action_embedding_layer(a) + timestep_embedding
        x = (
            # to (batch_size, 3, seq_length, embedding_dim)
            torch.stack(
                [return_embedding, state_embedding, action_embedding],
                dim=1,
            )
            # to (batch_size, seq_length, 3, embedding_dim)
            .permute(0, 2, 1, 3)
            # to (batch_size, seq_length * 3, embedding_dim)
            .reshape(batch_size, 3 * seq_length, self.embedding_dim)
        )
        x = self.layer_norm(x)

        # forward with gpt
        # to (batch_size, seq_length * 3, embedding_dim)
        x = self.gpt_model.forward(
            x,
            is_causal=True,
            mask=nn.Transformer.generate_square_subsequent_mask(seq_length * 3),
        )
        # to (batch_size, seq_length, 3, embedding_dim)
        x = x.reshape(batch_size, seq_length, 3, self.embedding_dim)
        # to (batch_size, 3, seq_length, embedding_dim)
        x = x.permute(0, 2, 1, 3)

        # action prediction
        action = self.action_predictor(x[:, 1])
        return action


class DecisionTransformerAlgo:
    def __init__(
        self,
        env: gym.Env,
        trajectories_path: list[Any],
        embedding_dim: int,
        context_length: int,
        num_gpt_layers: int,
        num_head: int,
        # optional
        lr: float = 1e-4,
        batch_size: int = 32,
        max_timesteps: int = 4096,
    ):
        self.env = env
        self.max_timesteps = max_timesteps
        self.model = DecisionTransformer(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            embedding_dim=embedding_dim,
            context_length=context_length,
            num_gpt_layers=num_gpt_layers,
            num_head=num_head,
            max_timesteps=max_timesteps,
        )
        norm_trajs, self.state_mean, self.state_std, self.reward_scale = (
            load_trajectories(trajectories_path)
        )
        self.dataset = DataLoader(
            DTDataset(norm_trajs, context_length),
            batch_size=batch_size,
            shuffle=True,
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr)
        self.loss_fn = nn.MSELoss()

    def learn(self, num_iter: int, expected_return: float) -> DecisionTransformer:
        save_path = Path(f"output/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        if save_path.exists() is False:
            save_path.mkdir(parents=True)

        tb = SummaryWriter(save_path / "tb_log")
        i = 0
        while True:
            for r, s, a, t in self.dataset:
                aa = self.model(r, s, a, t)
                loss = self.loss_fn(aa, a)
                tb.add_scalar("loss", loss, i)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.50)
                self.optimizer.step()
                if i % 100 == 0:
                    tb.add_scalar("episode_reward", self.evaluate(expected_return), i)
                if i % 1000 == 0 and i != 0:
                    self.save(save_path / f"model_{i}.pt")
                i += 1
            if i > num_iter:
                break
        return self.model

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def evaluate(self, expected_return: float) -> float:
        expected_return *= self.reward_scale
        returns = torch.zeros((1, self.max_timesteps, 1))
        states = torch.zeros(
            (
                1,
                self.max_timesteps,
                self.env.observation_space.shape[0],
            )
        )
        actions = torch.zeros(
            (
                1,
                self.max_timesteps,
                self.env.action_space.shape[0],
            )
        )
        timesteps = torch.arange(self.max_timesteps).unsqueeze(0)

        # give expected return
        returns[0, 0] = expected_return
        s, _ = self.env.reset()
        s = (s - self.state_mean) / self.state_std
        states[0, 0] = torch.from_numpy(s)
        total_r = 0
        t = 0
        while True:
            _start = max(0, t - self.model.context_length + 1)
            _end = _start + self.model.context_length
            _slice = slice(_start, _end)
            a = self.model.forward(
                returns[:, _slice],
                states[:, _slice],
                actions[:, _slice],
                timesteps[:, _slice],
            )[0, -1 if t >= self.model.context_length else t]
            s, r, t1, t2, _ = self.env.step(a.detach().numpy())
            s = (s - self.state_mean) / self.state_std
            total_r += r
            if t1 or t2:
                break
            else:
                actions[0, t] = a.detach()
                returns[0, t + 1] = returns[0, t] - r * self.reward_scale
                states[0, t + 1] = torch.from_numpy(s)
                t += 1

        return total_r
