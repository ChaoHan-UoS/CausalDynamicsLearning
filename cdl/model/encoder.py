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

        chemical_env_params = params.env_params.chemical_env_params
        self.hidden_objects_ind = chemical_env_params.hidden_objects_ind
        self.hidden_targets_ind = chemical_env_params.hidden_targets_ind
        self.hidden_ind = params.hidden_ind
        self.num_objects = params.env_params.chemical_env_params.num_objects
        self.num_colors = params.env_params.chemical_env_params.num_colors

        # d-set keys and observable keys
        self.keys_dset = chemical_env_params.keys_dset
        self.keys_remapped_dset = self.keys_dset + ['act'] + ['rew']
        self.keys = [key for key in params.obs_keys if params.obs_spec[key].ndim == 1]

        # input features of recurrent encoder, where d-set features are fed to LSTM
        self.feature_dim_p = np.sum([len(params.obs_spec[key]) for key in self.keys])
        self.continuous_state = params.continuous_state
        self.feature_inner_dim_p = None
        if not self.continuous_state:
            self.feature_inner_dim_p_dset = np.concatenate([params.obs_dims[key] for key in self.keys_dset])
            self.feature_inner_dim_p = np.concatenate([params.obs_dims[key] for key in self.keys])
        self.feature_inner_dim_remapped_p_dset = np.append(self.feature_inner_dim_p_dset, [params.action_dim, 1])

        # output features of recurrent encoder
        # self.feature_dim = len(params.keys_f)
        self.feature_dim = len(params.obs_keys_f)
        self.feature_inner_dim = None
        if not self.continuous_state:
            self.feature_inner_dim = np.concatenate([params.obs_dims_f[key] for key in params.obs_keys_f])
            # self.feature_inner_dim = np.concatenate([params.obs_dims_f[key] for key in params.keys_f])

        self.to(params.device)

    def forward(self, obs, s_0=None, info={}, detach=False):
        """
        :param obs: Batch(obs_i_key: (bs, stack_num, (n_pred_step), obs_i_shape))
        :return: [(bs, (n_pred_step), num_colors)] * (2 * num_objects)
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
            """FNN as identity map of observables"""
            obs_forward = [obs_k_i
                           for k in self.keys
                           for obs_k_i in torch.unbind(obs[k], dim=-1)]
            obs_forward = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
                           for obs_i, obs_i_dim in zip(obs_forward, self.feature_inner_dim_p)]

            obs_obs_forward = torch.stack(obs_forward, dim=0)  # (num_observables, bs, stack_num, (n_pred_step), num_colors)
            # slice the current obs from the stack_num history
            obs_obs_forward = torch.unbind(obs_obs_forward[:, :, -1])  # [(bs, (n_pred_step), num_colors)] * num_observables

            # identity encoder in the fully observable
            if len(self.hidden_ind) == 0:
                obs_enc = list(obs_obs_forward)  # [(bs, (n_pred_step), num_colors)] * (2 * num_objects)
                return obs_enc[:3], obs_enc[3:]

            """RNN fed by d-set"""
            obs = [obs_k_i
                   for k in self.keys_remapped_dset
                   for obs_k_i in torch.unbind(obs[k], dim=-1)]
            obs = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
                   for obs_i, obs_i_dim in zip(obs, self.feature_inner_dim_remapped_p_dset)]

            # slice in object and target observables from observation
            obs_obs = torch.stack(obs[:-2], dim=0)  # (num_dset_observables, bs, stack_num, (n_pred_step), num_colors)

            # slice in action and reward from observation
            obs_act, obs_rew = obs[-2], obs[-1]  # (bs, stack_num, (n_pred_step), action_dim / reward_dim)
            # mask out the last action / reward in the stacked obs
            obs_act[:, -1], obs_rew[:, -1] = obs_act[:, -1] * 0, obs_rew[:, -1] * 0

            obs_obs_dims = len(obs_obs.shape)
            if obs_obs_dims == 5:
                obs_obs = obs_obs.permute(1, 2, 0, 3, 4)  # (bs, stack_num, num_dset_observables, n_pred_step, num_colors)
                obs_obs_pred_step = [obs_obs_i.reshape(obs_obs.shape[:2] + (-1,))
                                     for obs_obs_i in
                                     torch.unbind(obs_obs, dim=-2)]  # [(bs, stack_num, num_dset_observables * num_colors)] * n_pred_step
                obs_act_pred_step = [obs_act_i
                                     for obs_act_i in
                                     torch.unbind(obs_act, dim=-2)]  # [(bs, stack_num, action_dim)] * n_pred_step
                obs_rew_pred_step = [obs_rew_i
                                     for obs_rew_i in
                                     torch.unbind(obs_rew, dim=-2)]  # [(bs, stack_num, reward_dim)] * n_pred_step
                # concatenate one-hot observables, action and scalar reward along the last dim
                # [(bs, stack_num, num_dset_observables * num_colors + action_dim + reward_dim)] * n_pred_step
                # obs_pred_step = [torch.cat((obses_i, acts_i, rews_i), dim=-1)
                #                  for obses_i, acts_i, rews_i in
                #                  zip(obs_obs_pred_step, obs_act_pred_step, obs_rew_pred_step)]
                # obs_pred_step = [torch.cat((obses_i, acts_i), dim=-1)
                #                  for obses_i, acts_i in
                #                  zip(obs_obs_pred_step, obs_act_pred_step)]
                obs_pred_step = obs_obs_pred_step
                obs_enc = []
                for obs_i in obs_pred_step:
                    obs_enc_i, s_n = super().forward(obs_i, s_0, info)  # obs_enc with shape (bs, logit_shape)
                    s_0 = s_n

                    obs_enc_i = obs_enc_i.reshape(-1, len(self.hidden_ind), self.num_colors)  # (bs, num_hidden_states, num_colors)
                    obs_enc_i = F.gumbel_softmax(obs_enc_i, hard=False) if self.training \
                        else F.one_hot(torch.argmax(obs_enc_i, dim=-1), obs_enc_i.size(-1)).float()
                    obs_enc.append(obs_enc_i)  # [(bs, num_hidden_states, num_colors)] * n_pred_step

                obs_enc = torch.stack(obs_enc, dim=-2)  # (bs, num_hidden_states, n_pred_step, num_colors)
            else:
                obs_obs = obs_obs.permute(1, 2, 0, 3)  # (bs, stack_num, num_dset_observables, num_colors)
                obs_obs = obs_obs.reshape(obs_obs.shape[:2] + (-1,))  # (bs, stack_num, num_dset_observables * num_colors)
                # concatenate one-hot observables, action and scalar reward along the last dim
                # (bs, stack_num, num_dset_observables * num_colors + action_dim + reward_dim)
                # obs = torch.cat((obs_obs, obs_act, obs_rew), dim=-1)
                # obs = torch.cat((obs_obs, obs_act), dim=-1)
                obs = obs_obs

                obs_enc, s_n = super().forward(obs, s_0, info)  # obs_enc with shape (bs, logit_shape)
                obs_enc = obs_enc.reshape(-1, len(self.hidden_ind), self.num_colors)  # (bs, num_hidden_states, num_colors)
                obs_enc = F.gumbel_softmax(obs_enc, hard=False) if self.training \
                    else F.one_hot(torch.argmax(obs_enc, dim=-1), obs_enc.size(-1)).float()

            obs_enc = torch.unbind(obs_enc, dim=1)  # [(bs, (n_pred_step), num_colors)] * num_hidden_states

            """concatenate outputs of FNN and RNN"""
            num_hidden_objects = len(self.hidden_objects_ind)
            num_obs_objects = self.num_objects - len(self.hidden_objects_ind)
            # [(bs, (n_pred_step), num_colors)] * (2 * num_objects)
            if len(self.hidden_targets_ind) > 0:
                obs_enc = obs_obs_forward[:self.hidden_objects_ind[0]] + obs_enc[:num_hidden_objects] \
                          + obs_obs_forward[self.hidden_objects_ind[0]:(num_obs_objects + self.hidden_targets_ind[0])] \
                          + obs_enc[num_hidden_objects:] + obs_obs_forward[(num_obs_objects + self.hidden_targets_ind[0]):]
            else:
                obs_enc = obs_obs_forward[:self.hidden_objects_ind[0]] + obs_enc[:num_hidden_objects] \
                          + obs_obs_forward[self.hidden_objects_ind[0]:]
            obs_enc = list(obs_enc)

            return obs_enc[:3], obs_enc[3:]


def obs_encoder(params):
    encoder_type = params.encoder_params.encoder_type
    if encoder_type == "recurrent":
        recurrent_enc_params = params.encoder_params.recurrent_enc_params
        chemical_env_params = params.env_params.chemical_env_params
        layer_num = recurrent_enc_params.layer_num
        # obs_shape = len(params.obs_keys) * chemical_env_params.num_colors + params.action_dim + chemical_env_params.max_steps + 1
        # obs_shape = len(chemical_env_params.keys_dset) * chemical_env_params.num_colors + params.action_dim + 1
        # obs_shape = len(chemical_env_params.keys_dset) * chemical_env_params.num_colors + params.action_dim
        obs_shape = len(chemical_env_params.keys_dset) * chemical_env_params.num_colors
        # logit_shape = chemical_env_params.num_objects * chemical_env_params.num_colors
        logit_shape = len(params.hidden_ind) * chemical_env_params.num_colors  # one-hot hidden states
        device = params.device
        hidden_layer_size = recurrent_enc_params.hidden_layer_size
        encoder = RecurrentEncoder(params, layer_num, obs_shape, logit_shape, device, hidden_layer_size).to(device)
        # print('\nTrainable parameters of the recurrent encoder')
        # for name, value in encoded_obs.named_parameters():
        #     print(name, value.shape)
    elif encoder_type == "identity":
        encoder = IdentityEncoder(params)
    else:
        raise ValueError("Unknown encoder_type: {}".format(encoder_type))

    return encoder


def act_encoder(actions_batch, env, params):
    """
    :param actions_batch: (bs, n_pred_step, action_dim)
    :return: (bs, n_pred_step, action_dim)
    """
    a_ids = actions_batch.view(-1).to(torch.int64)
    actions_batch_enc = torch.tensor([env.partial_act_dims[idx] for idx in a_ids],
                                     dtype=torch.int64, device=params.device)
    actions_batch_enc = actions_batch_enc.view(actions_batch.shape)

    return actions_batch_enc
