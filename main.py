# -*- coding: utf-8 -*-

# Kütüphanelerin eklenmesi
import os
import random
import gym
import pylab
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import argparse

# Kodlarda değişiklik yapmadan değişkenleri değiştirmek için konsol ekranında girdi alıyoruz.
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True, type=str, help="Env adını giriniz.",default="CartPole-v1")
parser.add_argument("--episode", required=True, type=int, help="İterasyon sayısını giriniz",default=1000)
parser.add_argument("--Save_Path", default="Models", type=str, help="Model kayıt adresi.")
parser.add_argument("--agent_type", default="ddqn", type=bool, help='True or False')
parser.add_argument("--test", type=bool, default=False, help='True or False')
parser.add_argument("--dueling", type=bool, default=True, help='True or False')

parser.add_argument("--memory", type=int, default=2000)
parser.add_argument("--learning_rate", type=float, default=0.00025)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epsilon", type=float, default=1.0)
parser.add_argument("--min_epsilon", type=float, default=0.01)
parser.add_argument("--decay_rate", type=float, default=0.999)
parser.add_argument("--gamma", type=float, default=0.95)

args = parser.parse_args()


# Modeli Oluşturuyoruz
def OurModel(input_shape, action_space, dueling):
    X_input = Input(input_shape)
    X = X_input

    # 512 nöron Girdi katmanı Environment Action sayısı aktivasyon fonksiyonu relu Ağırlık oluşturucu he_uniform
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X)
    # 256 nöron aktivasyon fonksiyonu relu Ağırlık oluşturucu he_uniform
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    # 64 nöron aktivasyon fonksiyonu relu Ağırlık oluşturucu he_uniform
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    if dueling:
        # state_value1 için
        state_value = Dense(1, kernel_initializer='he_uniform')(X)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)
        # state_value2 için        
        state_value2 = Dense(1, kernel_initializer='he_uniform')(X)
        state_value2 = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value2)
        # action_advantage1 için
        action_advantage = Dense(action_space, kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)
        # action_advantage2 için
        action_advantage2 = Dense(action_space, kernel_initializer='he_uniform')(X)
        action_advantage2 = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage2) 
        #      z  state_value ve action_advantage toplamı
        z = Add()([state_value, action_advantage])  
        # y  state_value2 ve action_advantage2 toplamı      
        y = Add()([state_value2, action_advantage2])   
        # X, z ve y nin toplamı
        X = Add()([z, y])
    else:
        # Çıkış sayısı action sayısı aktivasyon fonksiyonu relu Ağırlık oluşturucu he_uniform
        X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)
    # model tanımlanıyor.
    model = Model(inputs = X_input, outputs = X)
    # model oluşturuluyor.
    model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=args.learning_rate, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    # model ekrana yazdırılıyor.
    model.summary()
    return model


#Agent classı olusturduk
class DQNAgent:
    def __init__(self, env_name):
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.env.seed(0)  
        self.env._max_episode_steps = 4000
        #env kuralları
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.reward=[]

        self.EPISODES = args.episode
        self.memory = deque(maxlen=args.memory)
        
        self.gamma = args.gamma    # discount rate (indirim oranı)
        self.epsilon = args.epsilon  # exploration rate (keşif oranı)
        self.epsilon_min = args.min_epsilon # minimum exploration probability (minimum keşif oranı)
        self.epsilon_decay = args.decay_rate # exponential decay rate (epsilon azalma değeri)
        self.batch_size = args.batch_size 
        self.train_start = 1000

        # model parametreleri
        self.ddqn = args.agent_type # DDQN
        self.Soft_Update = False # Soft parametresi
        self.dueling = args.dueling # Dueling network

        self.TAU = 0.1 # Hedef model update parametresi

        self.Save_Path = args.Save_Path
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.scores, self.episodes, self.average = [], [], []
        
        # modele göre kaydedilen model ismi

        if self.ddqn:
            print("Double DQN")
            self.Model_name = os.path.join(self.Save_Path,"Dueling DDQN_"+self.env_name+".h5")
        else:
            print("DQN")
            self.Model_name = os.path.join(self.Save_Path,"Dueling DQN_"+self.env_name+".h5")
        
        # model ve hedef model
        self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size, dueling = self.dueling)
        self.target_model = OurModel(input_shape=(self.state_size,), action_space = self.action_size, dueling = self.dueling)

    # hedef model güncelleme
    def update_target_model(self):
        if not self.Soft_Update and self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        if self.Soft_Update and self.ddqn:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)
    # model hafızası
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                #epsilonu epsilon decayle çarpıyoruz
                self.epsilon *= self.epsilon_decay
    # action seçimi
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # hafızadan rastgele veri alınıyor
        minibatch = random.sample(self.memory, self.batch_size)

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []


        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # Ana ağı kullanarak başlangıç durumu için Q değerlerini tahmin ettik.
        target = self.model.predict(state)
        # Ana ağı kullanarak bitiş durumunda en iyi eylemi tahmin ettik. 
        target_next = self.model.predict(next_state)
        # Hedef ağı kullanarak durumu sonlandırmak için Q değerlerini tahmin ettil. 
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # kullanılan eylem için Q değerinde düzeltme.
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn: # Double - DQN
                    # Q Network eylemi seçer.
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # hedef Q Network, eylemi değerlendirir. 
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])   
                else: # Standard - DQN
                    # DQN, sonraki eylemler arasında maksimum Q değerini seçer 
                    # eylemin seçimi ve değerlendirilmesi hedef Q Ağı üzerindedir. 
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    # model yükleme
    def load(self, name):
        self.model.load_weights(name)
    #model kaydetme
    def save(self, name):
        self.model.save(name)
    #grafik oluşturma
    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        pylab.plot(self.episodes, self.average, 'r')
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        dqn = 'DQN_'
        softupdate = ''
        dueling = ''
        if self.ddqn: dqn = 'DDQN_'
        if self.Soft_Update: softupdate = '_soft'
        if self.dueling: dueling = '_Dueling'
        try:
            pylab.savefig(dqn+self.env_name+softupdate+dueling+".png")
        except OSError:
            pass

        return str(self.average[-1])[:5]
    # eğitim
    def run(self):
        for e in range(1,self.EPISODES+1):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    # model güncelleme
                    self.update_target_model()
                    
                    # grafik oluşturma
                    average = self.PlotModel(i, e)
                    print("episode: {}/{}, score: {}, e: {:.2}, average: {}".format(e, self.EPISODES, i, self.epsilon, average))

                    break
                self.replay()
                # model kaydetme
            if (e)%10==0:
                print("Saving trained model as", self.Model_name)
                self.save(self.Model_name)
# test
    def test(self):
        self.load(self.Model_name)
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break

if __name__ == "__main__":
    env_name = args.env
    agent = DQNAgent(env_name)
    if args.test==True:
        agent.test()
    else:
        agent.run()