import gymnasium as gym
import numpy as np
import time
from agent import Agent
import matplotlib.pyplot as plt

#Load the environment from Gymnasium
env = gym.make('CartPole-v1')

#Create the agent
agent = Agent(env.action_space, batch_size=32)

#Declare number of frames to train the agent
MAX_FRAMES = 5000

num_frames = 0
episodes = 0
scores = []

#Training loop
while num_frames < MAX_FRAMES:
    #Every episode, reset variables and the environment
    state, info = env.reset()
    done = False
    score = 0
    start = time.time()

    #Continue until the episode ends
    while not done:
        #Take action, and accept observations
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score+=reward

        #Store the transition in the relay buffer 
        agent.remember(state, action, reward, next_state, done)

        #Update state and model parameters, if applicable
        state = next_state
        agent.learn()

        num_frames+=1
    
    #Update episode data
    scores.append(score)
    avg_score = np.mean(scores[-25:]) #Take the average of the last 25 episodes
    episodes+=1
    
    #Print important information
    print(f"Episode: {episodes}, score: {score}, average_score: {avg_score}, epsilon: {agent.epsilon}, frame: {num_frames}, time: {time.time()-start}")

#Save model, if applicable
agent.save_model()

#Create x-axis
x = np.arange(episodes)

#Fit the trendline to the curve
coefficients = np.polyfit(x, scores, 4)
polynomial = np.poly1d(coefficients)

#Generate the y values of the trend line
trendline = polynomial(x)

#Plot the original data
plt.plot(scores, label='Scores')

#Plot the trend line and show the graph
plt.plot(x, trendline, label='Trend Line', linestyle='--', color='red')
plt.show()
