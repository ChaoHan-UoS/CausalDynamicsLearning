import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tianshou.utils.net.common import Recurrent, Net
from torch.distributions.one_hot_categorical import OneHotCategorical

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

class RewardEncoder(Net):
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params = params
        self.num_objects = params.env_params.chemical_env_params.num_objects
        self.num_hidden_objects = len(params.env_params.chemical_env_params.hidden_objects_ind)
        self.num_colors = params.env_params.chemical_env_params.num_colors

        self.keys = [key for key in params.obs_keys if params.obs_spec[key].ndim == 1]
        ####################### dset #######################
        self.keys_dset = ['obj0', 'obj1', 'obj3', 'obj4']
        self.keys_remapped_dset = self.keys_dset + ['act']
        ####################### dset #######################
        self.keys_remapped = self.keys + ['act']
        self.feature_dim_p = np.sum([len(params.obs_spec[key]) for key in self.keys])
        self.continuous_state = params.continuous_state
        self.feature_inner_dim_p = None
        if not self.continuous_state:
            self.feature_inner_dim_p_dset = np.concatenate([params.obs_dims[key] for key in self.keys_dset])
            self.feature_inner_dim_p = np.concatenate([params.obs_dims[key] for key in self.keys])
        self.feature_inner_dim_remapped_p_dset = np.append(self.feature_inner_dim_p_dset, params.action_dim)
        self.feature_inner_dim_remapped_p = np.append(self.feature_inner_dim_p, params.action_dim)

        # the output feature of the recurrent encoder
        self.feature_dim = self.num_objects
        self.feature_inner_dim = None
        if not self.continuous_state:
            self.feature_inner_dim = np.concatenate([params.obs_dims_f[key] for key in params.obs_keys_f])

        self.to(params.device)

    def forward(self, obs, s_0=None, info={}, detach=False):
        """
        :param obs: Batch(obs_i_key: (bs, stack_num, obs_i_shape)) or Batch(obs_i_key: (bs, stack_num, 1, obs_i_shape))
        :return: a list of num_objects elements, each of which is a tensor with shape (bs, num_colors) or (bs, 1, num_colors)
        """
        if self.continuous_state:
            # overwrite some observations for out-of-distribution evaluation
            if not getattr(self, "manipulation_train", True):
                test_scale = self.manipulation_test_scale
                obs = {k: torch.randn_like(v) * test_scale if "marker" in k else v
                       for k, v in obs.items()}
            obs = torch.cat([obs[k] for k in self.keys], dim=-1)
            return obs
        else:
            # obs_forward = [obs_k_i
            #        for k in self.keys_remapped
            #        for obs_k_i in torch.unbind(obs[k], dim=-1)]
            # obs_forward = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
            #        for obs_i, obs_i_dim in zip(obs_forward, self.feature_inner_dim_remapped_p)]
            # obs_obs_forward = torch.stack(obs_forward[:-2], dim=0)  # shape (num_observed_objects, bs, stack_num, (1), num_colors)
            # obs_obs_forward = torch.unbind(obs_obs_forward[:, :, -1])

            obs = [obs_k_i
                   for k in self.keys_remapped_dset
                   for obs_k_i in torch.unbind(obs[k], dim=-1)]
            obs = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
                   for obs_i, obs_i_dim in zip(obs, self.feature_inner_dim_remapped_p_dset)]
            obs_obs_forward = obs_obs = torch.stack(obs[:-1], dim=0)
            obs_obs_forward = torch.unbind(obs_obs_forward[:, :, -1])

            obs_obs_dims = len(obs_obs.shape)
            if obs_obs_dims == 5:
                obs_obs = obs_obs.permute(1, 2, 0, 3, 4)  # shape (bs, stack_num, num_observed_objects, 1, num_colors)
            else:
                obs_obs = obs_obs.permute(1, 2, 0, 3)  # shape (bs, stack_num, num_observed_objects, num_colors)
            obs_obs = obs_obs.reshape(obs_obs.shape[:2] + (-1,))  # shape (bs, stack_num, num_observed_objects * num_colors)
            obs_act = obs[-1]  # shape (bs, stack_num, (1), action_dim)
            obs_act[:, -1] = obs_act[:, -1] * 0  # mask out the last action in the stacked obs
            obs_act = obs_act.reshape(obs_act.shape[:2] + (-1,))  # shape (bs, stack_num, action_dim)
            # concatenate one-hot observation, action and episode step along the last dim
            obs = torch.cat([obs_obs, obs_act], dim=-1)  # shape (bs, stack_num, num_observed_objects * num_colors + action_dim)

            obs, s_n = super().forward(obs, s_0, info)  # obs with shape (bs, logit_shape)
            # overwrite some observations for out-of-distribution evaluation
            if not getattr(self, "chemical_train", True):
                assert self.params.env_params.env_name == "Chemical"
                assert self.params.env_params.chemical_env_params.continuous_pos
                test_scale = self.chemical_test_scale
                obs = [obs_i if obs_i.shape[-1] > 1 else torch.randn_like(obs_i) * test_scale for obs_i in obs]
            obs = obs.reshape(self.num_hidden_objects, -1, self.num_colors)  # shape (num_hidden_objects, bs, num_colors)
            if self.training:
                obs = F.gumbel_softmax(obs, hard=False)
            else:
                obs = F.one_hot(torch.argmax(obs, dim=-1), obs.size(-1)).float()
            if obs_obs_dims == 5:
                obs = obs.unsqueeze(dim=-2)
            # print('Recovered obj 2')
            # print(obs[:, 0])
            obs = obs_obs_forward[:2] + torch.unbind(obs) + obs_obs_forward[2:]  # concatenate RNN's output and FNN's
            # obs = torch.unbind(obs)  # for supervised RNN
            obs = list(obs)

            return obs

class IdentityEncoder(nn.Module):
    # extract 1D obs and concatenate them
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.keys = [key for key in params.obs_keys_f + params.goal_keys if params.obs_spec_f[key].ndim == 1]
        self.feature_dim = np.sum([len(params.obs_spec_f[key]) for key in self.keys])

        self.continuous_state = params.continuous_state
        self.feature_inner_dim = None
        if not self.continuous_state:
            self.feature_inner_dim = np.concatenate([params.obs_dims_f[key] for key in self.keys])

        self.to(params.device)

    def forward(self, obs, detach=False):
        obs = obs[:, -1]
        if self.continuous_state:
            # overwrite some observations for out-of-distribution evaluation
            if not getattr(self, "manipulation_train", True):
                test_scale = self.manipulation_test_scale
                obs = {k: torch.randn_like(v) * test_scale if "marker" in k else v
                       for k, v in obs.items()}
            obs = torch.cat([obs[k] for k in self.keys], dim=-1)
            return obs
        else:
            obs = [obs_k_i
                   for k in self.keys
                   for obs_k_i in torch.unbind(obs[k], dim=-1)]
            obs = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
                   for obs_i, obs_i_dim in zip(obs, self.feature_inner_dim)]
            # overwrite some observations for out-of-distribution evaluation
            if not getattr(self, "chemical_train", True):
                assert self.params.env_params.env_name == "Chemical"
                assert self.params.env_params.chemical_env_params.continuous_pos
                test_scale = self.chemical_test_scale
                obs = [obs_i if obs_i.shape[-1] > 1 else torch.randn_like(obs_i) * test_scale for obs_i in obs]

            # print('True obj 2')
            # print(obs[2][0], '\n')

            return obs[:len(self.params.obs_keys_f)], obs


class RecurrentEncoder(Recurrent):
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params = params
        self.num_objects = params.env_params.chemical_env_params.num_objects
        self.hidden_objects_ind = params.env_params.chemical_env_params.hidden_objects_ind
        self.num_hidden_objects = len(self.hidden_objects_ind)
        self.num_colors = params.env_params.chemical_env_params.num_colors

        # d-set
        self.keys_dset_ = params.env_params.chemical_env_params.keys_dset
        self.keys_dset = self.keys_dset_ + params.goal_keys
        self.keys_remapped_dset = self.keys_dset + ['act']

        self.keys_ = [key for key in params.obs_keys if params.obs_spec[key].ndim == 1]
        self.keys = self.keys_ + params.goal_keys
        self.keys_remapped = self.keys + ['act']

        # input features of recurrent encoder, where d-set features are fed to LSTM
        self.feature_dim_p = np.sum([len(params.obs_spec[key]) for key in self.keys])
        self.continuous_state = params.continuous_state
        self.feature_inner_dim_p = None
        if not self.continuous_state:
            self.feature_inner_dim_p_dset = np.concatenate([params.obs_dims[key] for key in self.keys_dset])
            self.feature_inner_dim_p = np.concatenate([params.obs_dims[key] for key in self.keys])
        self.feature_inner_dim_remapped_p_dset = np.append(self.feature_inner_dim_p_dset, params.action_dim)
        self.feature_inner_dim_remapped_p = np.append(self.feature_inner_dim_p, params.action_dim)

        # output features of recurrent encoder
        self.feature_dim = self.num_objects
        self.feature_inner_dim = None
        if not self.continuous_state:
            self.feature_inner_dim = np.concatenate([params.obs_dims_f[key] for key in params.obs_keys_f])

        self.to(params.device)

    def forward(self, obs, s_0=None, info={}, detach=False):
        """
        :param obs: Batch(obs_i_key: (bs, stack_num, obs_i_shape)) or Batch(obs_i_key: (bs, stack_num, 1, obs_i_shape))
        :return: a list of num_objects elements, each of which is a tensor with shape (bs, num_colors) or (bs, 1, num_colors)
        """
        if self.continuous_state:
            # overwrite some observations for out-of-distribution evaluation
            if not getattr(self, "manipulation_train", True):
                test_scale = self.manipulation_test_scale
                obs = {k: torch.randn_like(v) * test_scale if "marker" in k else v
                       for k, v in obs.items()}
            obs = torch.cat([obs[k] for k in self.keys], dim=-1)
            return obs
        else:
            obs_forward = [obs_k_i
                   for k in self.keys_remapped
                   for obs_k_i in torch.unbind(obs[k], dim=-1)]
            obs_forward = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
                   for obs_i, obs_i_dim in zip(obs_forward, self.feature_inner_dim_remapped_p)]
            obs_obs_forward_ = torch.stack(obs_forward[:-1], dim=0)  # shape (num_observed_objects, bs, stack_num, (1), num_colors)
            obs_obs_forward_ = torch.unbind(obs_obs_forward_[:, :, -1])  # slice the current obs from the stack_num history
            obs_obs_forward = obs_obs_forward_[:len(self.keys_)]
            obs_obs_forward_target = obs_obs_forward_[-len(self.params.goal_keys):]

            obs = [obs_k_i
                   for k in self.keys_remapped_dset
                   for obs_k_i in torch.unbind(obs[k], dim=-1)]
            obs = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
                   for obs_i, obs_i_dim in zip(obs, self.feature_inner_dim_remapped_p_dset)]
            obs_obs_ = torch.stack(obs[:-1], dim=0)
            obs_obs = obs_obs_[:len(self.keys_dset_)]

            obs_obs_dims = len(obs_obs.shape)
            if obs_obs_dims == 5:
                obs_obs = obs_obs.permute(1, 2, 0, 3, 4)  # shape (bs, stack_num, num_observed_objects, 1, num_colors)
            else:
                obs_obs = obs_obs.permute(1, 2, 0, 3)  # shape (bs, stack_num, num_observed_objects, num_colors)
            obs_obs = obs_obs.reshape(obs_obs.shape[:2] + (-1,))  # shape (bs, stack_num, num_observed_objects * num_colors)
            obs_act = obs[-1]  # shape (bs, stack_num, (1), action_dim)
            obs_act[:, -1] = obs_act[:, -1] * 0  # mask out the last action in the stacked obs
            obs_act = obs_act.reshape(obs_act.shape[:2] + (-1,))  # shape (bs, stack_num, action_dim)
            # concatenate one-hot observation, action and episode step along the last dim
            obs = torch.cat([obs_obs, obs_act], dim=-1)  # shape (bs, stack_num, num_observed_objects * num_colors + action_dim)

            obs, s_n = super().forward(obs, s_0, info)  # obs with shape (bs, logit_shape)
            # overwrite some observations for out-of-distribution evaluation
            if not getattr(self, "chemical_train", True):
                assert self.params.env_params.env_name == "Chemical"
                assert self.params.env_params.chemical_env_params.continuous_pos
                test_scale = self.chemical_test_scale
                obs = [obs_i if obs_i.shape[-1] > 1 else torch.randn_like(obs_i) * test_scale for obs_i in obs]
            obs = obs.reshape(self.num_hidden_objects, -1, self.num_colors)  # shape (num_hidden_objects, bs, num_colors)
            if self.training:
                obs = F.gumbel_softmax(obs, hard=False)
            else:
                obs = F.one_hot(torch.argmax(obs, dim=-1), obs.size(-1)).float()
            if obs_obs_dims == 5:
                obs = obs.unsqueeze(dim=-2)
            # print('Recovered obj 2')
            # print(obs[:, 0])
            # obs = obs_obs_forward[:1] + torch.unbind(obs) + obs_obs_forward[1:]  # concatenate RNN's output and FNN's
            obs = obs_obs_forward[:self.hidden_objects_ind[0]] + torch.unbind(obs) + obs_obs_forward[self.hidden_objects_ind[0]:]  # concatenate RNN's output and FNN's
            obs_target = obs + obs_obs_forward_target
            # obs = torch.unbind(obs)  # for supervised RNN
            obs, obs_target = list(obs), list(obs_target)

            return obs, obs_target


def make_encoder(params):
    encoder_type = params.encoder_params.encoder_type
    if encoder_type == "recurrent":
        recurrent_enc_params = params.encoder_params.recurrent_enc_params
        chemical_env_params = params.env_params.chemical_env_params
        layer_num = recurrent_enc_params.layer_num
        # obs_shape = len(params.obs_keys) * chemical_env_params.num_colors + params.action_dim + chemical_env_params.max_steps + 1
        obs_shape = len(chemical_env_params.keys_dset) * chemical_env_params.num_colors + params.action_dim
        # logit_shape = chemical_env_params.num_objects * chemical_env_params.num_colors
        logit_shape = len(chemical_env_params.hidden_objects_ind) * chemical_env_params.num_colors  # recover the hidden objects
        device = params.device
        hidden_layer_size = recurrent_enc_params.hidden_layer_size
        encoder = RecurrentEncoder(params, layer_num, obs_shape, logit_shape, device, hidden_layer_size)
        # print('\nTrainable parameters of the recurrent encoder')
        # for name, value in encoded_obs.named_parameters():
        #     print(name, value.shape)
    elif encoder_type == "identity":
        encoder = IdentityEncoder(params)
    else:
        raise ValueError("Unknown encoder_type: {}".format(encoder_type))

    return encoder
