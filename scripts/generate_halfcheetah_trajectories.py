import numpy as np
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from imitation.data import serialize


from stable_baselines3 import PPO

# Parallel environments
vec_env = make_vec_env(
    "HalfCheetah-v4",
    n_envs=8,
    rng=np.random.default_rng(42),
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
)
model = PPO("MlpPolicy", vec_env, verbose=1)

model.learn(total_timesteps=100_0000)
model.save("data/halfcheetah-sb3ppo-1m-steps")
trajs = rollout.rollout(
    model,
    vec_env,
    rollout.make_sample_until(min_episodes=1000),
    rng=np.random.default_rng(42),
)
serialize.save("data/halfcheetah-sb3ppo-1m-steps", trajs)

model.learn(total_timesteps=100_0000, reset_num_timesteps=False)
model.save("data/halfcheetah-sb3ppo-2m-steps")
trajs = rollout.rollout(
    model,
    vec_env,
    rollout.make_sample_until(min_episodes=1000),
    rng=np.random.default_rng(42),
)
serialize.save("data/halfcheetah-sb3ppo-2m-steps", trajs)
