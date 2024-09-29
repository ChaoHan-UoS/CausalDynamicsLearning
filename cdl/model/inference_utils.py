import numpy as np
from collections import deque, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gumbel import gumbel_sigmoid


def reset_layer(w, b=None):
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing w with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features))
    # nn.init.kaiming_uniform_(w, a=np.sqrt(5))
    nn.init.kaiming_uniform_(w, nonlinearity='relu')
    if b is not None:
        fan_in = w.shape[1]
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(b, -bound, bound)


def reparameterize(mu, log_std):
    std = torch.exp(log_std)
    eps = torch.randn_like(std)
    return eps * std + mu


def forward_network(input, weights, biases, activation=F.relu, dropout=nn.Dropout(p=0)):
    """
    given an input and a multi-layer networks (i.e., a layer-wise list of weights and a list of biases),
        apply the network to the input, and return an output
    the same activation function is applied to all layers except for the last layer
    """
    x = input
    for i, (w, b) in enumerate(zip(weights, biases)):
        # x (p_bs, bs, in_dim), bs: data batch size which must be 1D
        # w (p_bs, in_dim, out_dim), p_bs: parameter batch size
        # b (p_bs, 1, out_dim)
        x = torch.bmm(x, w) + b  # (p_bs, bs, out_dim)
        if i < len(weights) - 1 and activation:
            x = F.layer_norm(x, normalized_shape=(x.shape[-1],))
            x = activation(x)
            x = dropout(x)
    return x


def forward_network_batch(inputs, weights, biases, activation=F.relu, dropout=nn.Dropout(p=0)):
    """
    given a list of inputs and a list of ONE-LAYER networks (i.e., a list of weights and a list of biases),
        apply each network to each input, and return a list
    """
    x = []
    for x_i, w, b in zip(inputs, weights, biases):
        # x_i (p_bs, bs, in_dim), bs: data batch size which must be 1D
        # w (p_bs, in_dim, out_dim), p_bs: parameter batch size
        # b (p_bs, 1, out_dim)
        x_i = torch.bmm(x_i, w) + b  # (p_bs, bs, out_dim)
        if activation:
            x_i = F.layer_norm(x_i, normalized_shape=(x_i.shape[-1],))
            x_i = activation(x_i)
            x_i = dropout(x_i)
        x.append(x_i)
    return x


def forward_gated_network(input, weights, biases, gate_weights, gate_biases, deterministic=False, activation=F.relu):
    """
    given an input and a multi-layer networks (i.e., a list of weights and a list of biases),
        apply the network to each input, and return output
    the same activation function is applied to all layers except for the last layer
    """
    gate = None
    if len(gate_weights):
        gate_log_alpha = input
        for i, (w, b) in enumerate(zip(gate_weights, gate_biases)):
            # gate_log_alpha (bs, p_bs, in_dim), bs: data batch size
            # w (p_bs, out_dim, in_dim), p_bs: parameter batch size
            # b (p_bs, out_dim)
            gate_log_alpha = gate_log_alpha.unsqueeze(dim=-2)  # (bs, p_bs, 1, in_dim)
            gate_log_alpha = (gate_log_alpha * w).sum(dim=-1) + b  # (bs, p_bs, out_dim)
            if i < len(gate_weights) - 1 and activation:
                gate_log_alpha = activation(gate_log_alpha)

        if deterministic:
            gate = (gate_log_alpha > 0).float()
        else:
            gate = gumbel_sigmoid(gate_log_alpha, device=gate_log_alpha.device, hard=True)

    x = input
    for i, (w, b) in enumerate(zip(weights, biases)):
        # x (bs, p_bs, in_dim), bs: data batch size
        # w (p_bs, out_dim, in_dim), p_bs: parameter batch size
        # b (p_bs, out_dim)
        x = x.unsqueeze(dim=-2)  # (bs, p_bs, 1, in_dim)
        x = (x * w).sum(dim=-1) + b  # (bs, p_bs, out_dim)
        if i < len(weights) - 1 and activation:
            x = activation(x)
        if i == len(weights) - 2 and gate is not None:
            x = x * gate
    return x


def get_controllable(mask):
    feature_dim = mask.shape[0]
    M = mask[:, :feature_dim]
    I = mask[:, feature_dim:]

    # feature that are directly affected by actions
    action_children = []
    for i in range(feature_dim):
        if I[i].any():
            action_children.append(i)

    # decedents of those features
    controllable = []
    queue = deque(action_children)
    while len(queue):
        feature_idx = queue.popleft()
        controllable.append(feature_idx)
        for i in range(feature_dim):
            if M[i, feature_idx] and (i not in controllable) and (i not in queue):
                queue.append(i)
    return controllable


def get_state_abstraction(mask):
    feature_dim = mask.shape[0]
    M = mask[:, :feature_dim]

    controllable = get_controllable(mask)
    # ancestors of controllable features
    action_relevant = []
    queue = deque(controllable)
    while len(queue):
        feature_idx = queue.popleft()
        if feature_idx not in controllable:
            action_relevant.append(feature_idx)
        for i in range(feature_dim):
            if (i not in controllable + action_relevant) and (i not in queue):
                if M[feature_idx, i]:
                    queue.append(i)

    abstraction_idx = list(set(controllable + action_relevant))
    abstraction_idx.sort()

    abstraction_graph = OrderedDict()
    for idx in abstraction_idx:
        abstraction_graph[idx] = [i for i, e in enumerate(mask[idx]) if e]

    return abstraction_graph


def obs_batch2tuple(obs, params):
    """
    :param obs: Batch(obs_i_key: (bs, seq_len, obs_i_shape))
    :return: oa_tuple: [tuple(tuple('obs_i_key', (obs_i_value,)))] * (bs * (seq_len - 2))
    """
    obs_v = obs[next(iter(dict(obs)))]
    bs, seq_len = obs_v.shape[:2]

    # convert to a list of tuples for hashing, with len(list) = bs * (seq_len -2)
    # {obs_i_key: (bs * (seq_len - 2), obs_i_shape)}; t=1 to T-2
    oa = {k: obs[k][:, :-2].reshape(-1, obs[k].shape[-1]) for k in params.obs_keys_f + ['act'] if k in obs}
    oa_tuple = [tuple((k, (v[i].item(),)) for k, v in oa.items())
                for i in range(bs * (seq_len - 2))]
    return oa_tuple


def hidden_batch2tuple(h):
    """
    :param h: (bs, seq_len, num_hiddens * num_colors)
    :return: h_tuple: [tuple(num_hiddens * num_colors,)] * (bs * (seq_len -2))
    """
    # (bs * (seq_len - 2), num_hiddens * num_colors); t=1 to T-2
    h = h[:, 1:-1].reshape(-1, h.shape[-1])
    h_tuple = [tuple(h[i].tolist()) for i in range(h.shape[0])]
    return h_tuple


def count2prob(counter):
    """
    :param counter: Counter({key: count})
    :return: probs: {key: prob}
    """
    total_counts = sum(counter.values())
    probs = {t: count / total_counts for t, count in counter.items()}
    return probs


def prob2llh(probs, keys):
    """
    :param probs: {key_i: prob_i}
    :param keys: [key] * len
    :return: (len, ) tensor as likelihood
    """
    return torch.tensor([probs[k] for k in keys])


def rm_dup(tup_list):
    # Dictionary to keep track of seen tuples and their indices
    seen = {}
    dups = []

    # Identify indices of duplicate tuples
    for idx, item in enumerate(tup_list):
        if item in seen:
            dups.append(idx)
        else:
            seen[item] = idx

    # mask in unique tuples
    mask = [i for i in range(len(tup_list)) if i not in dups]
    uni_tups = [tup_list[i] for i in mask]

    return mask, uni_tups