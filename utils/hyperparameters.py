BATCH_SIZE = 128 # number of transitions sampled from the replay buffer
GAMMA = 0.99 # discount factor
EPS_START = 0.9 # Starting value of epsilon
EPS_END = 0.05 # End value of epsilon
EPS_DECAY = 1000 # Epsilon decay rate
TAU = 0.005 # Update rate for the target network
LR = 3e-4 # Learning rate for the Adam optimizer