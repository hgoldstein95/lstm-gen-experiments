# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bst import *

torch.manual_seed(1)

d_in = 11
d_hidden = 11

def num_as_onehot(n):
    ret = torch.zeros(1, d_in)
    ret[0, n] = 1.0

    return ret

def choice_as_vec(choice):
    if isinstance(choice, ChooseLeaf):
        num = 0
    else:
        num = choice.value + 1

    return num_as_onehot(num)

choice_as_vec(ChooseNode(5))

# input_choices is n by 11, where n is the number of choices in the sequence
def eval(input_choices, lstm, fc):
    input_onehot = tuple(choice.to_tensor().unsqueeze(0).float() for choice in input_choices)

    inputs_stacked = torch.stack(input_onehot, 0)
    hidden_0 = torch.zeros(1, 1, d_hidden)
    cell_0 = torch.zeros(1, 1, d_hidden)
    
    out, (hidden_end, cell_end) = lstm(inputs_stacked, (hidden_0, cell_0))

    output_logit = fc(hidden_end).squeeze(0).squeeze(0)

    output_probs = F.softmax(output_logit, 0)
    
    return output_probs

def compute_loss(input_choices, probs):
    fitnesses = torch.stack([
        torch.tensor(
            fitness(parse_choices(input_choices + [choice])[-1]),
            dtype = torch.float,
        )
        for choice in Choice.all_choices()
	])

    return -torch.dot(probs, fitnesses)

lstm = nn.LSTM(d_in, d_hidden)
fc = nn.Linear(d_hidden, d_in)
all_params = tuple(lstm.parameters()) + tuple(fc.parameters())

optimizer = optim.SGD(all_params, lr=0.1)

all_tests = []
all_failures = []
for epoch_num in range(100):
    lstm.zero_grad()
    fc.zero_grad()
    epoch_loss = 0
    tests = []
    failures = 0
    for trial_run in range(100):
        so_far = [ChooseNode(5)]
        trial_loss = 0
        while True: # Build a new test
            probs = eval(so_far, lstm, fc)
            trial_loss += compute_loss(so_far, probs)
            so_far.append(Choice.sample_from_tensor(probs))
            partial = parse_choices(so_far)[-1]
            if partial.is_complete():
                tests.append(partial)
                break
            if not partial.is_bst():
                failures += 1
                break
        
        epoch_loss += trial_loss

    print(epoch_loss)
    all_tests.append(tests)
    all_failures.append(failures)
    epoch_loss.backward()
    optimizer.step()

all_tests
all_failures