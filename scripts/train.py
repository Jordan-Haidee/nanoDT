from dataclasses import dataclass

import gymnasium as gym
import tyro

from decision_transformer.model import DecisionTransformerAlgo


@dataclass
class Args:
    env: str = "HalfCheetah-v4"  # Environment name
    trajectories_path: str = "data/halfcheetah-medium-v2"  # Trajectory dataset path
    train_iters: int = 20000  # Number of training iterations
    expected_return: int = 6000  # Expected return
    num_head: int = 4  # Number of heads in the multi-head attention
    embedding_dim: int = 128  # Embedding dimension
    context_length: int = 20  # DT context length
    num_gpt_layers: int = 3  # Number of GPT layers
    lr: float = 1e-4  # Learning rate
    batch_size: int = 64  # Batch size


def train(args):
    env = gym.make(args.env)
    dt_algo = DecisionTransformerAlgo(
        env,
        trajectories_path=args.trajectories_path,
        embedding_dim=args.embedding_dim,
        context_length=args.context_length,
        num_gpt_layers=args.num_gpt_layers,
        num_head=args.num_head,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    dt_algo.learn(args.train_iters, expected_return=args.expected_return)


args = tyro.cli(Args)
train(args)
