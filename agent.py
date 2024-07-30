import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from collections import deque
import numpy as np
import random

class Agent():
    
    def __init__(self, 
                 action_space, 
                 batch_size, 
                 lr=1e-3, 
                 discount_factor=0.95, 
                 epsilon=1, 
                 epsilon_min=0.1, 
                 epsilon_decay=0.99, 
                 max_buffer_size=100000, 
                 save_file=None
                 ):
        super().__init__()

        #Intatiate model parameters
        self.lr = lr
        self.batch_size = batch_size
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_buffer = deque(maxlen=max_buffer_size)
        self.save_file = save_file

        #Build and assign the DQN
        self.dqn = self.build_model()
    
    #Builds and returns the DQN
    def build_model(self):
        #Create sequential model
        model=tf.keras.Sequential([
            Input(shape=(4,)),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(self.action_space.n, activation="linear")
        ])

        #Compile with optimizer and loss function
        model.compile(optimizer=Adam(learning_rate=self.lr), loss="mean_squared_error")
        return model
    
    #Adds a transition into the replay buffer
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    #Takes a state and returns an action
    def take_action(self, state):
        #Agent takes a random action with a probability of epsilon
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_space.n)
        
        state = np.array([state])
        q_values = self.dqn.predict(state, verbose=0)

        return np.argmax(q_values)
    
    #Gets a random batch of size number_of_samples from the buffer
    def sample_buffer(self, number_of_samples):
        return random.sample(self.replay_buffer, number_of_samples)

    #Multiplies the epsilon value by epsilon_decay
    def adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    #Trains the model on one batch of data
    def learn(self):
        #Do not start learning if there are not enough samples in the replay buffer
        if len(self.replay_buffer) < self.batch_size: return

        #Take a random sample from the buffer and extract the contents
        batch = self.sample_buffer(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        #Convert to NumPy arrays
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        #Predict the Q-values of the current and next states, then use those values to train the model
        q_prediction = self.dqn.predict(states, verbose=0)
        q_next_prediction = self.dqn.predict(next_states, verbose=0)

        q_target = np.copy(q_prediction)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + (1 - dones) * self.discount_factor * np.max(q_next_prediction, axis=1)

        self.dqn.train_on_batch(states, q_target)

        #Reduce epsilon after training
        self.adjust_epsilon()

    #Saves the model to save_file, given save_file is not none
    def save_model(self):
        if self.save_file:
            self.dqn.save(self.save_file)

    #Loads the model from save_file, given save_file is not none
    def load_model(self):
        if self.save_file:
            self.dqn = load_model(self.save_file)
