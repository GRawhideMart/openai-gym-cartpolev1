import torch
from memory import Transition
from utils.hyperparameters import BATCH_SIZE, GAMMA
from device import standard as device

class DQLearner:
    """
    A deep Q-learning agent that uses a policy network and a target network to learn to interact with an environment. The agent stores transitions in a replay memory and optimizes the policy network using these transitions.
    """
    def __init__(self, memory, policy_network, target_network):
        """
        Initializes the DQLearner object.
        
        Parameters:
        memory (ReplayMemory): the replay memory used by the agent to store transitions
        policy_network (nn.Module): the policy network used by the agent to make decisions
        target_network (nn.Module): the target network used by the agent to stabilize training of the policy network
        
        Returns:
        None
        """
        self.memory = memory
        self.policy_network = policy_network
        self.target_network = target_network

    def calculate(self):
        """
        Calculates the Q values, expected Q values, and V values for a batch of transitions from the replay memory.
        
        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: a tuple containing the Q values, expected Q values, and V values
        """
        batch = self._sample_transitions()
        state_b, action_b, non_final_next_states_b, reward_b = self._single_batches(batch)
        state_action_values = self._q_values(states=state_b, actions=action_b)
        next_state_values = self._v_values(batch=batch, non_final_next_states=non_final_next_states_b)
        expected_state_action_values = self._expected_q_values(reward_b, next_state_values)
        return state_action_values, next_state_values, expected_state_action_values

    def _sample_transitions(self):
        """
        Samples a batch of transitions from the agent's replay memory.
        
        Returns:
        Transition: a batch of transitions
        """
        transitions = self.memory.sample(BATCH_SIZE)
        return Transition(*zip(*transitions))
    
    def _single_batches(self, batch):
        """
        Extracts and concatenates the corresponding elements of a batch of transitions.
        
        Parameters:
        batch (Transition): a batch of transitions
        
        Returns:
        Tuple[torch.Tensor]: a tuple containing the concatenated state, action, next state, and reward tensors
        """
        return (
            torch.cat(batch.state),
            torch.cat(batch.action),
            torch.cat([s for s in batch.next_state if s is not None]),
            torch.cat(batch.reward)
        )
    
    def _non_final_mask(self, batch):
        """
        Computes a mask indicating which states in a batch of transitions are non-final.
        
        Parameters:
        batch (Transition): a batch of transitions
        
        Returns:
        torch.Tensor: a boolean tensor representing the mask
        """
        return torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            dtype=torch.bool,
            device=device
        )

    def _q_values(self, states, actions):
        """
        Computes the Q values for a batch of states and actions.
        
        Parameters:
        states (torch.Tensor): a tensor of states
        actions (torch.Tensor): a tensor of actions
        
        Returns:
        torch.Tensor: a tensor of Q values
        """
        return self.policy_network(states).gather(1, actions)

    def _expected_q_values(self, rewards, next_state_values):
        """
        Computes the expected Q values for a batch of rewards and next state values.
        
        Parameters:
        rewards (torch.Tensor): a tensor of rewards
        next_state_values (torch.Tensor): a tensor of next state values
        
        Returns:
        torch.Tensor: a tensor of expected Q values
        """
        return rewards + (GAMMA * next_state_values)

    def _v_values(self, batch, non_final_next_states):
        """
        Computes the V values for a batch of non-final next states.
        
        Parameters:
        batch (Transition): a batch of transitions
        non_final_next_states (torch.Tensor): a tensor of non-final next states
        
        Returns:
        torch.Tensor: a tensor of V values
        """
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        non_final_mask = self._non_final_mask(batch=batch)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
        
        return next_state_values

    