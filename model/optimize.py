import torch
import torch.nn as nn
from memory import Transition
from device import standard as device

def optimize_model(optimizer, memory, policy_net, target_net, BATCH_SIZE, GAMMA):
    """
    Optimize the policy network using a batch of transitions from the memory object.
    
    Parameters:
    optimizer (torch.optim.Optimizer): the optimizer used to update the policy network
    memory (ReplayMemory): a replay memory object containing a collection of transitions
    policy_net (torch.nn.Module): the policy network to be optimized
    target_net (torch.nn.Module): a target network used to compute expected state-action values
    BATCH_SIZE (int): the batch size to use for optimization
    GAMMA (float): the discount factor
    
    Returns:
    None
    """

    # Until memory is not long enough to be sampled from, the agent
    # should just keep observing the environment and remembering.
    if len(memory) < BATCH_SIZE:
        return

    # Sample some elements from memory. This will produce a variable like
    # [Transition1, Transition2, ...] i.e. [(state_1, action_1, next_state_1, reward_1), ...]
    # I actually need Transition([state_1, action_1,...], ...)
    transitions = memory.sample(BATCH_SIZE)

    # zip is the inverse of itself. Hence, this operation turns a batch
    # array of transitions to a transition of batch arrays
    batch = Transition(*zip(*transitions))

    # I need a way of checking if the state is the final one: for this,
    # I will define a boolean tensor through the map function, then turn
    # it into a tensor
    non_final_mask = tuple(map(lambda s: s is not None, batch.next_state))
    non_final_mask = torch.tensor(non_final_mask, device=device, dtype=torch.bool)

    # Next, I need to extract to concatenate the same elements of the batch together
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net.
    # Note that policy_net is a model, so calling it as a function will actually
    # fire its forward method. So, effectively, the next line starts the 
    # forward propagation using the state batch, generating an output tensor.
    # The gather method then, along the columns (1-index) subsets the actions.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    # Compute the expected Q values
    expected_state_action_values = reward_batch + (GAMMA * next_state_values)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()