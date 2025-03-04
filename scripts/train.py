import gymnasium as gym

from decision_transformer.model import DecisionTransformerAlgo

env = gym.make("HalfCheetah-v4")

dt_algo = DecisionTransformerAlgo(
    env,
    trajectories_path="data/halfcheetah-sb3ppo-1m-steps",
    embedding_dim=128,
    context_length=20,
    num_gpt_layers=3,
    num_head=1,
    lr=1e-4,
    batch_size=64,
)

dt_algo.learn(20000, expected_return=6000)
