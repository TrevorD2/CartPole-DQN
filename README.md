# Cart Pole Deep Q-Network
A reinforcement learning agent employing a Deep Q-Network for the Cart Pole environment. Hyperparameters should be tuned to match individual performance requirements; additionally, the architecture of the DQN can be easily modified as well.

This is a short and simple implementation of a reinforcement learning agent, which can be modified and adapted to solve other reinforcement learning problems. This model is trained in the Cart Pole environment provided by OpenAI's Gymnasium library. The agent takes in four values as the input for the state (cart position, cart velocity, pole angle, and pole  angular velocity), and outputs Q-values for moving the cart. More information on the Cart Pole environment can be found in the documentation: https://gymnasium.farama.org/environments/classic_control/cart_pole/

The network's simple architecture consists of several dense layers, which can be expanded on to fit performance needs, followed by an output layer with a size equal to the size of the action space. Even after training for a small number of episodes, the model demonstrates significant growth on the Cart Pole task. Performance can be improved even further if the model is trained on more frames.

![CartPole2Data](https://github.com/user-attachments/assets/6a1e1124-5e11-453e-b29c-3a3835bef454)
