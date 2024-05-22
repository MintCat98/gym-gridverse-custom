import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# refactored code from homework : dqn lab 

def get_network(num_states, num_actions) -> nn.Module:
    return nn.Sequential(
        nn.Linear(num_states, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, num_actions), 
        nn.Softmax()
    )

def get_action(state, network:nn.Module, epsilon=0.):
    """
    sample actions with epsilon-greedy policy
    recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
    """
    state = torch.tensor(state[None], dtype=torch.float32)
    q_values = network(state).detach().numpy()

    selected_action = None
    p = np.random.rand()
    if p < epsilon:
        selected_action = np.random.randint(len(q_values[0]))
    else:
        selected_action = np.argmax(q_values)
    return int(selected_action)


def compute_td_loss(states, actions, rewards, next_states, is_done, network:nn.Module, gamma=0.99, check_shapes=False):
    """ Compute td loss using torch operations only. Use the formula above. """
    states = torch.tensor(
        states, dtype=torch.float32)                                  # shape: [batch_size, state_size]
    actions = torch.tensor(actions, dtype=torch.long)                 # shape: [batch_size]
    rewards = torch.tensor(rewards, dtype=torch.float32)              # shape: [batch_size]

    next_states = torch.tensor(next_states, dtype=torch.float32)      # shape: [batch_size, state_size]
    is_done = torch.tensor(is_done, dtype=torch.uint8)                # shape: [batch_size]

    # get q-values for all actions in current states
    predicted_qvalues = network(states)                               # shape: [batch_size, n_actions]

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[                # shape: [batch_size]
      range(states.shape[0]), actions
    ]

    # compute q-values for all actions in next states
    predicted_next_qvalues = network(next_states).detach()

    # compute V*(next_states) using predicted next q-values
    next_state_values = V_next_state = torch.max(predicted_next_qvalues, 0)[0]
    assert next_state_values.dtype == torch.float32

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    target_qvalues_for_actions = rewards + gamma * predicted_next_qvalues[0][np.argmax(predicted_next_qvalues)] #???
    # reward + gamma * max for a' in Q_(s', a')

    # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    target_qvalues_for_actions = torch.where(
        is_done, rewards, target_qvalues_for_actions)

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions.detach()) ** 2)
    if check_shapes:
        assert predicted_next_qvalues.data.dim(
        ) == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim(
        ) == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim(
        ) == 1, "there's something wrong with target q-values, they must be a vector"
    
    return loss


# ------- notice ---------

# must set optimizer in the control loop
# opt = torch.optim.Adam(network.parameters(), lr=1e-4)
