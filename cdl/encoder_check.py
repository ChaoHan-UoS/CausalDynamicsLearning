import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical

# hyper parameters
total_steps = 100000
bs = 128
lr = 0.001
adj_mat = torch.tensor([[1., 0., 0., 0., 0.],
                        [1., 1., 0., 0., 0.],
                        [1., 1., 1., 0., 0.],
                        [1., 1., 1., 1., 0.],
                        [1., 1., 1., 1., 1.]])

# adj_mat = torch.tensor([[1., 0., 0., 0., 0.],
#                         [1., 1., 0., 0., 0.],
#                         [0., 1., 1., 0., 0.],
#                         [0., 0., 1., 1., 0.],
#                         [0., 0., 0., 1., 1.]])
num_colors = 5
num_objs = adj_mat.size(0)
adj_mat_b = adj_mat.unsqueeze(0).repeat(bs, 1, 1)

# # lower_tri5
# # input mask for the hidden obj1
# mask_obs_h = torch.tensor([1., 0., 1., 1., 1.])
# mask_next_obs = torch.tensor([0., 0., 0., 0., 1.])
# # input mask for the observed obj0
# mask_obs_o0 = torch.tensor([0., 0., 1., 1., 1.])
# # input mask for the observed obj2
# mask_obs_o2 = torch.tensor([1., 0., 0., 1., 1.])
# # input mask for the observed obj3
# mask_obs_o3 = torch.tensor([1., 0., 1., 0., 1.])
# # input mask for the observed obj4
# mask_obs_o4 = torch.tensor([1., 0., 1., 1., 0.])

# lower_tri5
# input mask for the hidden obj1
mask_obs_h = torch.tensor([1., 0., 1., 0., 0.])
mask_next_obs = torch.tensor([0., 0., 1., 0., 0.])
# input mask for the observed obj0
mask_obs_o0 = torch.tensor([0., 0., 1., 0., 0.])
# input mask for the observed obj2
mask_obs_o2 = torch.tensor([1., 0., 0., 0., 0.])

# # chain5
# mask_obs_h = torch.tensor([0., 0., 1., 0., 0.])
# mask_next_obs = torch.tensor([0., 0., 1., 0., 0.])
# # input mask for the observed obj2
# mask_obs_o2 = torch.tensor([0., 0., 0., 0., 0.])

mlp_dims = [num_objs * (2 * num_colors + 1), 4 * num_objs, num_colors]


class Encoder(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for i in range(1, len(dims)):
            self.layers.append(nn.Linear(dims[i-1], dims[i]))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for i, l in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = OneHotCategorical(logits=l(x))
            else:
                x = F.relu(l(x))
        return x


encoder = Encoder(mlp_dims)
optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

# training loop
for step in range(total_steps):
    s = torch.randint(num_colors, (bs, num_objs)).float()                                # (bs, num_objs)
    a = torch.randint(num_objs, (bs,))                                                   # (bs,)
    a_onehot = F.one_hot(a, num_objs).float()                                              # (bs, num_objs)
    next_s = torch.fmod(torch.matmul(adj_mat_b, s.unsqueeze(-1)).squeeze(-1) + a_onehot, num_colors)     # (bs, num_objs)

    s_onehot = F.one_hot(s.long(), num_colors).float()  # (bs, num_objs, num_colors)
    s_h_masked = s_onehot * mask_obs_h.view(1, -1, 1)  # (bs, num_objs, num_colors)
    s_h_masked = s_h_masked.reshape((bs, -1))  # (bs, num_objs * num_colors)
    next_s_onehot = F.one_hot(next_s.long(), num_colors).float()  # (bs, num_objs, num_colors)
    next_s_h_masked = next_s_onehot * mask_next_obs.view(1, -1, 1)  # (bs, num_objs, num_colors)
    next_s_h_masked = next_s_h_masked.reshape((bs, -1))  # (bs, num_objs * num_colors)
    h_inp = torch.cat((s_h_masked, next_s_h_masked, a_onehot), dim=-1)  # (bs, num_objs * (num_colors * 2 + 1))
    h_dist = encoder(h_inp)  # (bs, num_colors)
    h_onehot = F.gumbel_softmax(h_dist.logits, hard=True)

    s_o0_masked = s_onehot * mask_obs_o0.view(1, -1, 1)  # (bs, num_objs, num_colors)
    s_o0_masked = list(torch.unbind(s_o0_masked, dim=1))  # [(bs, num_colors)] * num_objs
    s_o0_masked[1] = h_onehot
    s_o0_masked = torch.stack(s_o0_masked, dim=1)  # (bs, num_objs, num_colors)
    s_o0_masked = s_o0_masked.reshape((bs, -1))  # (bs, num_objs * num_colors)
    o0_inp = torch.cat((s_o0_masked, next_s_h_masked, a_onehot), dim=-1)  # (bs, num_objs * (num_colors * 2 + 1))
    o0_dist = encoder(o0_inp)  # (bs, num_colors)
    o0_onehot_label = s_onehot[:, 0]  # (bs, num_colors)
    nll_o0 = -o0_dist.log_prob(o0_onehot_label).mean()

    s_o2_masked = s_onehot * mask_obs_o2.view(1, -1, 1)  # (bs, num_objs, num_colors)
    s_o2_masked = list(torch.unbind(s_o2_masked, dim=1))  # [(bs, num_colors)] * num_objs
    s_o2_masked[1] = h_onehot
    s_o2_masked = torch.stack(s_o2_masked, dim=1)  # (bs, num_objs, num_colors)
    s_o2_masked = s_o2_masked.reshape((bs, -1))  # (bs, num_objs * num_colors)
    o2_inp = torch.cat((s_o2_masked, next_s_h_masked, a_onehot), dim=-1)  # (bs, num_objs * (num_colors * 2 + 1))
    o2_dist = encoder(o2_inp)  # (bs, num_colors)
    o2_onehot_label = s_onehot[:, 2]  # (bs, num_colors)
    nll_o2 = -o2_dist.log_prob(o2_onehot_label).mean()

    # s_o3_masked = s_onehot * mask_obs_o3.view(1, -1, 1)  # (bs, num_objs, num_colors)
    # s_o3_masked = list(torch.unbind(s_o3_masked, dim=1))  # [(bs, num_colors)] * num_objs
    # s_o3_masked[1] = h_onehot
    # s_o3_masked = torch.stack(s_o3_masked, dim=1)  # (bs, num_objs, num_colors)
    # s_o3_masked = s_o3_masked.reshape((bs, -1))  # (bs, num_objs * num_colors)
    # o3_inp = torch.cat((s_o3_masked, next_s_h_masked, a_onehot), dim=-1)  # (bs, num_objs * (num_colors * 2 + 1))
    # o3_dist = encoder(o3_inp)  # (bs, num_colors)
    # o3_onehot_label = s_onehot[:, 3]  # (bs, num_colors)
    # nll_o3 = -o3_dist.log_prob(o3_onehot_label).mean()

    # s_o4_masked = s_onehot * mask_obs_o4.view(1, -1, 1)  # (bs, num_objs, num_colors)
    # s_o4_masked = list(torch.unbind(s_o4_masked, dim=1))  # [(bs, num_colors)] * num_objs
    # s_o4_masked[1] = h_onehot
    # s_o4_masked = torch.stack(s_o4_masked, dim=1)  # (bs, num_objs, num_colors)
    # s_o4_masked = s_o4_masked.reshape((bs, -1))  # (bs, num_objs * num_colors)
    # o4_inp = torch.cat((s_o4_masked, next_s_h_masked, a_onehot), dim=-1)  # (bs, num_objs * (num_colors * 2 + 1))
    # o4_dist = encoder(o4_inp)  # (bs, num_colors)
    # o4_onehot_label = s_onehot[:, 4]  # (bs, num_colors)
    # nll_o4 = -o4_dist.log_prob(o4_onehot_label).mean()

    # backwards
    nll = nll_o0 + nll_o2
    # nll = nll_o0 + nll_o2 + nll_o3 + nll_o4
    # nll = nll_o2
    optimizer.zero_grad()
    nll.backward()
    optimizer.step()

    if (step + 1) % 100 == 0:
        print(f'step {step + 1}/{total_steps}, loss = {nll.item():.4f}')