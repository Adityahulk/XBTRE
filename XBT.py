import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
from btgym import BTgymEnv
import IPython.display as Display
import PIL.Image as Image
from gym import spaces

!git clone https://github.com/Kismuz/btgym.git

!cd btgym

!pip install -e .


from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d mczielinski/bitcoin-historical-data
!unzip bitcoin-historical-data.zip

df = pd.read_csv('/content/bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv')


df.isnull().sum(axis = 0)

df[1241713:1241718]

df.head()

df_n = df.dropna()

df_n.index = np.arange(0,3330541)

df.to_csv('bitcoin.csv')

import gym
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque

class DeepQNetwork:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=2000)
        
        self.gamma = 0.97     #dicounted rate is basically for getting future expected return
        self.epsilon = 1.0             #Initializing epsilon and its decay rate for epsilon greedy strategy//
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005     #learning_rate that is used while updating Qvalue in bellman's equation//
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=10, verbose=0)

    def target_train(self):                                                    
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


def main():
    env     = BTgymEnv(filename='bitcoin.csv') 
    gamma   = 0.9
    epsilon = .95

    trials  = 100
    trial_len = 1000

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials): 
        #dqn_agent.model= load_model("./model.model")
        cur_state = np.array(list(env.reset().items())[0][1])
        cur_state= np.reshape(cur_state, (30,4,1))
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            reward = reward*10 if not done else -10
            new_state =list(new_state.items())[0][1]
            new_state= np.reshape(new_state, (30,4,1))
            dqn_agent.target_train() # iterates target model

            cur_state = new_state
            if done:
                break
        
        print("Completed trial #{} ".format(trial))
        dqn_agent.render_all_modes(env)
        dqn_agent.save_model("model.model".format(trial))
        

if __name__ == "__main__":
    main()


