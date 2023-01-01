# Cart-Pole v1 problem from OpenAI Gym
Welcome to the openai-gym-cartpolev1 repository! This repository contains a reinforcement learning agent that is trained to balance a pole on a cart using the OpenAI Gym environment. The agent is implemented using the Q-learning algorithm, which allows it to learn and improve its performance through trial and error.

## Installation
To install and run this repository, you will need to have [conda](https://docs.conda.io/en/latest/) installed on your machine.  
1. Clone this repository to your local machine using `git clone https://github.com/GRawhideMart/openai-gym-cartpolev1.git`
2. Navigate to the repository directory using `cd openai-gym-cartpolev1`
3. Create a new conda environment using `conda create --name openai-gym-cartpolev1`
4. Activate the new environment using `conda activate openai-gym-cartpolev1`
5. Install the required packages using `conda install --file=requirements.txt`

## Agent and Memory
The agent has a memory which stores the states, actions, and rewards that it experiences as it interacts with the environment. This memory is used to update the agent's Q-values, which represent the expected reward for taking a particular action in a given state. The agent uses these Q-values to decide which action to take in each state, with the goal of maximizing its overall reward.

To run the agent, use the command `python agent.py`. This will train the agent and display its progress as it learns to balance the pole.

Thank you for visiting this repository! I hope you find it useful in your own exploration of reinforcement learning.