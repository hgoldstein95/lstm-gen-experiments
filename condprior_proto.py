# this is exploring the idea that we can decompose p(x | phi(x)) into a distribution over
# actions/AST nodes, and then directly approximating that distribution over actions via
# supervised learning or MCTS

import more_itertools
import itertools
import random
from bst import *

def all_2_partitions(iterable):
    il = list(iterable)
    return itertools.chain(
        [[il, []]],
        more_itertools.set_partitions(il, 2)
    )

def count_bsts_v2(elts):
    ret = 1
    for e in elts:
        l = count_bsts_v2({e2 for e2 in elts if e2 < e})
        r = count_bsts_v2({e2 for e2 in elts if e2 > e})

        ret += l * r
    return ret

#def count_bts_v2_helper(exact_elts):
#    if len(exact_elts) == 0:
#        return 1
#
#    ret = 0
#
#    for i, e in enumerate(exact_elts):
#        except_e = exact_elts[:i] + exact_elts[i+1:]
#        for p1, p2 in all_2_partitions(except_e):
#            l = count_bts_v2_helper(p1)
#            r = count_bts_v2_helper(p2)
#            ret += l * r
#
#    return ret
#
#def count_bts_v2(elts):
#    ret = 0
#
#    for p1, p2 in all_2_partitions(elts):
#        ret += count_bts_v2_helper(p1)
#
#    return ret

def first_hole_bounds(bt, lower_excl, upper_excl):
    if isinstance(bt, Leaf):
        return None, None
    elif isinstance(bt, Node):
        l_l, l_r = first_hole_bounds(bt.left, lower_excl, bt.value)

        if l_l is not None:
            return l_l, l_r
        
        return first_hole_bounds(bt.right, bt.value, upper_excl)
    elif isinstance(bt, Hole):
        return lower_excl, upper_excl
    else:
        assert False

def first_hole_replace(bt, v):
    if isinstance(bt, Leaf):
        assert False
    elif isinstance(bt, Node):
        if not bt.left.is_complete():
            return Node(
                first_hole_replace(bt.left, v),
                bt.value,
                bt.right,
            )
        if not bt.right.is_complete():
            return Node(
                bt.left,
                bt.value,
                first_hole_replace(bt.right, v),
            )
        assert False
    elif isinstance(bt, Hole):
        return v
    else:
        assert False

def count_completions(
    t: Tree, minimum: int = MIN_VAL, maximum: int = MAX_VAL
) -> float:
    if isinstance(t, Leaf):
        return 1

    elif isinstance(t, Node):
        l = count_completions(t.left, minimum=minimum, maximum=t.value - 1)
        r = count_completions(t.right, minimum=t.value + 1, maximum=maximum)
        return l * r

    else:  # isinstance(t, Hole):
        return count_bsts(maximum - minimum + 1)

def sample_action(bt, lower_excl, upper_excl):
    l, u = first_hole_bounds(bt, lower_excl, upper_excl)

    choices = []
    for i in range(l + 1, u):
        bt2 = first_hole_replace(bt, Node(Hole(), i, Hole()))
        choices.append(bt2)

    choices.append(first_hole_replace(bt, Leaf()))

    choices_weight = [count_completions(bt2, lower_excl + 1, upper_excl - 1) for bt2 in choices]
    #print("\n".join(repr(p) for p in zip(choices, choices_weight)))

    return random.choices(choices, choices_weight)[0]

def sample_tree(lower_excl, upper_excl):
    cur = Hole()
    ret = []
    while not cur.is_complete():
        cur = sample_action(cur, lower_excl, upper_excl)
        ret.append(cur)
    
    return ret


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#class MyNN(nn.Module):
#    def __init__(self, d_in, d_hidden, d_out):
#        super(MyNN,self).__init__()
#        self.fc_hole_0 = nn.Linear(d_in, d_hidden)
#        self.fc_hole_1 = nn.Linear(d_hidden, d_hidden)
#        self.fc_hole_2 = nn.Linear(d_hidden, d_hidden)
#        self.fc_hole_3 = nn.Linear(d_hidden, d_out)
#
#
#    def forward(self, minimum, maximum):
#        #input = torch.tensor([[minimum, maximum]], dtype = torch.float)
#        input = torch.tensor([[maximum - minimum + 1]], dtype = torch.float)
#        return (
#            self.fc_hole_3(
#            self.fc_hole_2(
#            self.fc_hole_1(
#            self.fc_hole_0(
#            input
#        )))))

d_hidden = 1
nn_count_bsts_approx = nn.Sequential(
    nn.Linear(1, 1)
)
def count_bsts_approx(minimum, maximum):
    #input = torch.tensor([[minimum, maximum]], dtype = torch.float)
    input = torch.tensor([[maximum - minimum + 1]], dtype = torch.float)
    return nn_count_bsts_approx(input)

optimizer = optim.SGD(nn_count_bsts_approx.parameters(), lr=0.001)

for epoch_num in range(100):
    print("====== EPOCH ======")
    nn_count_bsts_approx.zero_grad()
    epoch_loss = 0
    for minimum in range(11):
        for maximum in range(minimum, 11):
            outp = count_bsts_approx(minimum, maximum)
            gt = math.log(count_bsts(maximum - minimum + 1))
            gt = maximum - minimum - 1
            loss = torch.abs(outp - gt)
            #print(maximum - minimum + 1, outp.item(), gt, loss.item())

            loss.backward()
            epoch_loss += loss.clone().detach()

    print(epoch_loss)
    optimizer.step()

# TODO: get SGD working at all
# TODO: supervised learning on a "policy"? (output 11-dimensional vector, take softmax, minimize KLD vs the actual marginals)