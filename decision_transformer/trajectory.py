from dataclasses import field
from imitation.data.types import TrajectoryWithRew
from imitation.data import serialize
from torch import Tensor
import torch
import dataclasses
import numpy as np
import random
from torch.utils.data import Dataset


class DTTrajectory(TrajectoryWithRew):
    returns: Tensor = field(init=False)
    states: Tensor = field(init=False)
    actions: Tensor = field(init=False)
    timesteps: Tensor = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        # compute returns from rewards
        self.returns = torch.from_numpy(self.rews).flipud().cumsum(0).flipud().float()
        self.states = torch.from_numpy(self.obs).float()
        self.actions = torch.from_numpy(self.acts).float()
        self.timesteps = torch.arange(len(self.returns)).long()

    def __len__(self) -> int:
        return len(self.returns)


class DTDataset(Dataset):
    trajectories: list[DTTrajectory]

    def __init__(self, trajectories: list[TrajectoryWithRew], context_length: int):
        self.trajectories = [
            DTTrajectory(traj.obs, traj.acts, None, traj.terminal, traj.rews)
            for traj in trajectories
        ]
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        traj = self.trajectories[idx]
        _start = random.randint(0, len(traj) - self.context_length)
        _end = _start + self.context_length
        _slice = slice(_start, _end)
        return (
            traj.returns[_slice].unsqueeze(-1),
            traj.states[_slice],
            traj.actions[_slice],
            traj.timesteps[_slice],
        )


def load_trajectories(
    path: str,
) -> tuple[list[TrajectoryWithRew], np.ndarray, np.ndarray, float]:
    # load trajectories
    trajs = serialize.load(path)
    # compute reward scale
    reward_scale = 1 / (
        10
        ** int(
            np.log10(
                np.max([traj.rews.sum() for traj in trajs]),
            )
        )
    )
    # normalize states and rewards
    all_states = np.stack([traj.obs for traj in trajs]).reshape(
        -1, trajs[0].obs.shape[-1]
    )
    mean, std = all_states.mean(0), all_states.std(0) + 1e-6
    res = []
    for traj in trajs:
        res.append(
            dataclasses.replace(
                traj,
                obs=(traj.obs - mean) / std,
                rews=traj.rews * reward_scale,
            )
        )
    return res, mean, std, reward_scale
