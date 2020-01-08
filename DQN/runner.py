# Standard library
from collections import namedtuple
from itertools import count
import time

# User made files
from EnvManager import EnvironmentManager
from Memory import ReplayMemory
from DQN import DQN
from Agent import Agent
from GreedyStrat import EpsilonGreedyStrategy

# Non standard Libraries

from tensorflow import keras
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

def actionRecreator():
    dicty = {0: np.array([-1.0, 1.0, 0]), 1: np.array([-0.75, 1.0, 0]), 2: np.array([-0.5, 1.0, 0]), 3: np.array([-0.25, 1.0, 0]), 4: np.array([0.0, 1.0, 0]), 5: np.array([0.25, 1.0, 0]),
             6: np.array([0.5, 1.0, 0]), 7: np.array([0.75, 1.0, 0]), 8: np.array([1.0, 1.0, 0]), 9: np.array([-1.0, 0.5, 0]), 10: np.array([0.0, 0.5, 0]), 11: np.array([1.0, 0.5, 0]), 12: np.array([0.0, 0.0, 1.0])}
    return dicty

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))
    try:
        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.action)
        t3 = torch.cat(batch.reward)
        t4 = torch.cat(batch.next_state)

    except Exception as e:
        print(batch.reward)
        print(e)

    return (t1,t2,t3,t4)

def update(memory, policy_net):
    global BATCH_SIZE, GAMMA, Experience, device, optimizer
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Experience(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def resize(img):
    img = img.cpu().detach().numpy()
    mat = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    return cv2.resize(mat, dsize=(224, 224))

# Global variable initializer
Experience = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actionDict = actionRecreator()
num_episodes = 50000
max_timestep = 1000

# Hyper Parameter initializer
n_latent_var = 64
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
learningrate = 0.001

# Creates the environment, search strategy and agent
env = EnvironmentManager(device, "CarRacing-v0", actionDict)
strat = EpsilonGreedyStrategy(EPS_END, EPS_END, EPS_DECAY)
agent = Agent(strat, env.num_actions_available(), device)

# Creates the policy and target network
policy_net = DQN(env.get_screen_height(), env.get_screen_width(), env.num_actions_available(), n_latent_var).to(device)
target_net = DQN(env.get_screen_height(), env.get_screen_width(), env.num_actions_available(), n_latent_var).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learningrate)
memory = ReplayMemory(10000)

InputLayer = keras.layers.Input(batch_shape=(None, 224, 224, 3))
road = keras.applications.MobileNetV2(input_tensor=InputLayer, weights=None, classes=2)
Nadam = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
road.compile(optimizer=Nadam, loss='mean_squared_error', metrics=['accuracy'])
road.load_weights('Unitygym.h5')
print("Loaded keras weights")

writer = open("DQNRoad.csv", mode="a")

def runner(num_episodes, max_timestep, BATCH_SIZE, env):
    episodeRew = []
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        env.done = False

        episodeRew.append(0)

        last_screen = env.get_state()
        current_screen = env.get_state()

        state = current_screen - last_screen

        episodeStart = time.time()
        timestep = 0

        while not env.done:
            timestep += 1
            action = agent.select_action(state, policy_net)
            reward = env.step(action)
            reward = reward - 0.00001
            stateResize = resize(state)
            stateResize = np.resize(stateResize, new_shape=(1, 224, 224, 3))
            prediction = road.predict(stateResize)
            if np.argmax(prediction) != 1:
                reward = reward - 0.001
            reward = torch.tensor(reward).to('cuda')
            episodeRew[-1] += reward.float()
            #print(reward)
            next_state = env.get_state()
            memory.push(Experience(torch.tensor(state), torch.tensor(action), torch.tensor(next_state), torch.tensor(reward.float())))
            state = next_state
            # env.render('rgb_array')
            if timestep == max_timestep:
                env.done = True


        if memory.can_provide_sample(BATCH_SIZE):
            experiences = memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * GAMMA) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Perform one step of the optimization (on the target network)
            # update(memory, policy_net)

        print("Episode {}, gave reward: {} and took {} seconds for {} timesteps".format(i_episode, episodeRew[-1], (time.time()-episodeStart), timestep))
        #writer.write(str(episodeRew) + "\n")
        #print("Written")
        if i_episode%100+1 == 0:
            for a in range(len(100)):
                writer.write(str(episodeRew[a]) + ",")
            writer.write("\n")
            episodeRew = []
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            #torch.save(policy_net.state_dict(), "Log/Carracing_DQN_lr{}_{}.pt".format(str(learningrate), i_episode))
            target_net.load_state_dict(policy_net.state_dict())
        if i_episode % 100 == 0:
            torch.save(policy_net.state_dict(), "Log/DQN_road_{}.pt".format(i_episode))
    env.close()
    writer.close()

runner(num_episodes, max_timestep, BATCH_SIZE, env)

