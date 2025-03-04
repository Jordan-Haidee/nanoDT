import numpy as np
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data import serialize


from stable_baselines3 import PPO

# Parallel environments
vec_env = make_vec_env("Pendulum-v1", n_envs=8)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100_0000)
model.save("data/pendulum-sb3ppo")


env = make_vec_env(
    "Pendulum-v1",
    n_envs=8,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
)
expert = load_policy(
    "ppo",
    venv=env,
    path="data/pendulum-sb3ppo",
)
trajs = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_episodes=5000),
    rng=np.random.default_rng(42),
)


serialize.save("data/pendulum-sb3ppo", trajs)
