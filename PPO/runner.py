import gym

import numpy as np
#from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

env = gym.make('CarRacing-v0')

totSteps = 50000
n_actions = env.action_space.shape[-1]
param_noise=None
#action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = PPO1(MlpPolicy, env, verbose=1)
#model.learn(total_timesteps=totSteps)
done = False
a = 0
obs = env.reset()

while not done:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    a+= 1
    if a >= totSteps:
        done=True
