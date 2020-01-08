import gym

import numpy as np
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines import DDPG
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
import cv2

from tensorflow import keras

def resize(img):
    img = img
    mat = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    return cv2.resize(mat, dsize=(224, 224))


env = gym.make('CarRacing-v0')

totSteps = 50000
episodeSteps = 1000
n_actions = env.action_space.shape[-1]
param_noise=None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
#model.learn(total_timesteps=totSteps)

InputLayer = keras.layers.Input(batch_shape=(None, 224, 224, 3))
road = keras.applications.MobileNetV2(input_tensor=InputLayer, weights=None, classes=2)
Nadam = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
road.compile(optimizer=Nadam, loss='mean_squared_error', metrics=['accuracy'])
road.load_weights('Unitygym.h5')
print("Loaded keras weights")

writer = open("DDPG_Road.csv", mode="a")
episodeRew = []
for episode in range(totSteps):
    dones = False
    a = 0
    episodeRew.append(0)
    obs = env.reset()
    while not dones:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #episodeRew += rewards
        a+= 1
        stateResize = resize(obs)
        stateResize = np.resize(stateResize, new_shape=(1, 224, 224, 3))
        prediction = road.predict(stateResize)
        if np.argmax(prediction) != 1:
            rewards = rewards - 0.001
            #reward = torch.tensor(reward).to('cuda')

        if a == episodeSteps:
            done=True
        episodeRew[-1] += rewards

    print("Episode: {}, gave reward {}".format(episode, episodeRew[-1]))
    if episode % 100 == 0 and episode != 0:
        for a in range(100):
            writer.write(str(episodeRew[a]) + ",")
        writer.write("\n")
        episodeRew = []
        model.save("Logs/DDPG_Road{}".format(episode))
        print("Model saved")

writer.close()
