import sys

from collections import OrderedDict, Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical

from model.inference_utils import (reset_layer, forward_network, forward_network_batch,
                                   obs_batch2tuple, hidden_batch2tuple, count2prob, prob2llh, rm_dup)

def reset_layer_lstm(w, b=None):
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features))
    nn.init.kaiming_uniform_(w, a=np.sqrt(5))
    # nn.init.kaiming_uniform_(w, nonlinearity='relu')
    if b is not None:
        fan_in = w.shape[1]
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(b, -bound, bound)


class LSTMCell(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens

        # initialize weights with a normal distribution and biases with zeros
        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(num_hiddens, num_inputs),
                          init_weight(num_hiddens, num_hiddens),
                          nn.Parameter(torch.zeros(num_hiddens)))
        self.w_xi, self.w_hi, self.b_i = triple()  # Input gate
        self.w_xf, self.w_hf, self.b_f = triple()  # Forget gate
        self.w_xo, self.w_ho, self.b_o = triple()  # Output gate
        self.w_xc, self.w_hc, self.b_c = triple()  # Input node
        self.reset_params()

    def reset_params(self):
        reset_layer_lstm(self.w_xi, self.b_i)
        reset_layer_lstm(self.w_hi)
        reset_layer_lstm(self.w_xf, self.b_f)
        reset_layer_lstm(self.w_hf)
        reset_layer_lstm(self.w_xo, self.b_o)
        reset_layer_lstm(self.w_ho)
        reset_layer_lstm(self.w_xc, self.b_c)
        reset_layer_lstm(self.w_hc)

    def forward(self, x, s_0=None):
        """
        :param x: (bs, num_inputs)
        :param s_0: ((bs, (bs, num_hiddens))) * 2
        :return: h: (bs, num_hiddens)
                 c: (bs, num_hiddens)
        """
        if s_0 is None:
            h_0 = torch.zeros((x.shape[0], self.num_hiddens), device=x.device)  # (bs, num_hiddens)
            c_0 = torch.zeros((x.shape[0], self.num_hiddens), device=x.device)
        else:
            h_0, c_0 = s_0
        i = torch.sigmoid(torch.matmul(x, self.w_xi.T) +
                          torch.matmul(h_0, self.w_hi) + self.b_i)
        f = torch.sigmoid(torch.matmul(x, self.w_xf.T) +
                          torch.matmul(h_0, self.w_hf) + self.b_f)
        o = torch.sigmoid(torch.matmul(x, self.w_xo.T) +
                          torch.matmul(h_0, self.w_ho) + self.b_o)
        c_tilde = torch.tanh(torch.matmul(x, self.w_xc.T) +
                             torch.matmul(h_0, self.w_hc) + self.b_c)
        c = f * c_0 + i * c_tilde  # (bs, num_hiddens)
        c = F.layer_norm(c, normalized_shape=(self.num_hiddens,))
        h = o * torch.tanh(c)  # (bs, num_hiddens)
        return h, c


class RNN(nn.Module):
    def __init__(self, params, obs_shape, hidden_layer_size, layer_num, logit_shape):
        super().__init__()
        dropout = params.encoder_params.recurrent_enc_params.dropout
        self.device = params.device
        self.obs_shape = obs_shape
        self.hidden_layer_size = hidden_layer_size

        self.fc1 = nn.Linear(obs_shape, hidden_layer_size)
        self.lstmcell1 = LSTMCell(hidden_layer_size, hidden_layer_size)
        self.lstmcell2 = LSTMCell(hidden_layer_size, hidden_layer_size)
        # self.layer_norm = nn.LayerNorm(hidden_layer_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_layer_size, logit_shape)

        # apply the same initialization scheme to linear layers in the encoder
        # as those in the transition model
        # reset_layer(self.fc1.weight, self.fc1.bias)
        # reset_layer(self.fc2.weight, self.fc2.bias)

    def forward(self, obs, s_0=None):
        """
        :param obs: (bs, stack_num, obs_shape)
        :param s_0: ((layer_num, bs, hidden_layer_size)) * 2
        :return: logit: (bs, logit_shape)
                 h_n: (layer_num, bs, hidden_layer_size)
                 c_n: (layer_num, bs, hidden_layer_size)
        """
        bs = obs.shape[0]
        seq_len = obs.shape[1]
        obs = self.fc1(obs)  # (bs, stack_num, hidden_layer_size)
        if s_0 is None:
            h_l0 = torch.zeros(bs, self.hidden_layer_size).to(self.device)
            c_l0 = torch.zeros(bs, self.hidden_layer_size).to(self.device)
            h_l1 = torch.zeros(bs, self.hidden_layer_size).to(self.device)
            c_l1 = torch.zeros(bs, self.hidden_layer_size).to(self.device)
        else:
            h_0, c_0 = s_0  # (layer_num, bs, hidden_layer_size)
            h_l0, c_l0 = h_0[0], c_0[0]  # (bs, hidden_layer_size)
            h_l1, c_l1 = h_0[1], c_0[1]
        for t in range(seq_len):
            # first lstm layer
            h_l0, c_l0 = self.lstmcell1(obs[:, t], (h_l0, c_l0))  # (bs, hidden_layer_size)
            # h_l0 = self.layer_norm(h_l0)
            h_l0 = self.dropout(h_l0)

            # second lstm layero
            h_l1, c_l1 = self.lstmcell2(h_l0, (h_l1, c_l1))
            # h_l1 = self.layer_norm(h_l1)
            h_l1 = self.dropout(h_l1)
        logit = self.fc2(h_l1)  # (bs, logit_shape)
        h_n = torch.stack((h_l0, h_l1), dim=0)  # (layer_num, bs, hidden_layer_size)
        c_n = torch.stack((c_l0, c_l1), dim=0)  # (layer_num, bs, hidden_layer_size)
        return logit, (h_n.detach(), c_n.detach())


class RecurrentEncoder(RNN):
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)

        chemical_env_params = params.env_params.chemical_env_params
        self.hidden_objects_ind = chemical_env_params.hidden_objects_ind
        self.hidden_targets_ind = chemical_env_params.hidden_targets_ind
        self.hidden_ind = params.hidden_ind
        self.num_objects = params.env_params.chemical_env_params.num_objects
        self.num_colors = params.env_params.chemical_env_params.num_colors

        # d-set keys and all observable keys
        self.keys_dset = params.env_params.chemical_env_params.keys_dset
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
        self.feature_dim = len(params.obs_keys_f)
        self.feature_inner_dim = None
        if not self.continuous_state:
            self.feature_inner_dim = np.concatenate([params.obs_dims_f[key] for key in params.obs_keys_f])

    def forward(self, obs, s_0=None):
        """
        :param obs: Batch(obs_i_key: (bs, stack_num, (n_pred_step), obs_i_shape))
        :return: [(bs, (n_pred_step), num_colors)] * num_objects, (full states)
                 [(bs, (n_pred_step), num_colors)] * num_objects  (full target states)
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

            obs_obs_forward = torch.stack(obs_forward,
                                          dim=0)  # (num_observables, bs, stack_num, (n_pred_step), num_colors)
            # slice the current obs from the stack_num history
            obs_obs_forward = torch.unbind(
                obs_obs_forward[:, :, -1])  # [(bs, (n_pred_step), num_colors)] * num_observables

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
            obs_act, obs_rew = obs[-2].clone(), obs[
                -1].clone()  # (bs, stack_num, (n_pred_step), action_dim / reward_dim)
            # mask out the last action / reward in the stacked obs
            # obs_act[:, -1], obs_rew[:, -1] = obs_act[:, -1] * 0, obs_rew[:, -1] * 0
            obs_act[:, -1], obs_rew = obs_act[:, -1] * 0, obs_rew * 0
            # obs_act, obs_rew = obs_act * 0, obs_rew * 0

            obs_obs_dims = len(obs_obs.shape)
            if obs_obs_dims == 5:
                obs_obs = obs_obs.permute(1, 2, 0, 3, 4)  # (bs, stack_num, num_dset_objects, n_pred_step, num_colors)
                obs_obs_pred_step = [obs_obs_i.reshape(obs_obs.shape[:2] + (-1,))
                                     for obs_obs_i in
                                     torch.unbind(obs_obs,
                                                  dim=-2)]  # [(bs, stack_num, num_dset_observables * num_colors)] * n_pred_step
                obs_act_pred_step = [obs_act_i
                                     for obs_act_i in
                                     torch.unbind(obs_act, dim=-2)]  # [(bs, stack_num, action_dim)] * n_pred_step
                obs_rew_pred_step = [obs_rew_i
                                     for obs_rew_i in
                                     torch.unbind(obs_rew, dim=-2)]  # [(bs, stack_num, reward_dim)] * n_pred_step

                # concatenate one-hot observables, action and scalar reward along the last dim
                # [(bs, stack_num, num_dset_observables * num_colors + action_dim + reward_dim)] * n_pred_step
                obs_pred_step = [torch.cat((obses_i, acts_i, rews_i), dim=-1)
                                 for obses_i, acts_i, rews_i in
                                 zip(obs_obs_pred_step, obs_act_pred_step, obs_rew_pred_step)]

                obs_enc = []
                for obs_i in obs_pred_step:
                    obs_enc_i, s_n = super().forward(obs_i, s_0)  # obs_enc with shape (bs, logit_shape)
                    # s_0 = s_n

                    obs_enc_i = obs_enc_i.reshape(-1, len(self.hidden_ind),
                                                  self.num_colors)  # (bs, num_hidden_states, num_colors)
                    obs_enc_i = F.gumbel_softmax(obs_enc_i, hard=True) if self.training \
                        else F.one_hot(torch.argmax(obs_enc_i, dim=-1), obs_enc_i.size(-1)).float()
                    obs_enc.append(obs_enc_i)  # [(bs, num_hidden_states, num_colors)] * n_pred_step
                obs_enc = torch.stack(obs_enc, dim=-2)  # (bs, num_hidden_states, n_pred_step, num_colors)
            else:
                obs_obs = obs_obs.permute(1, 2, 0, 3)  # (bs, stack_num, num_dset_observables, num_colors)
                obs_obs = obs_obs.reshape(
                    obs_obs.shape[:2] + (-1,))  # (bs, stack_num, num_dset_observables * num_colors)

                # concatenate one-hot observables, action and scalar reward along the last dim
                # (bs, stack_num, num_dset_observables * num_colors + action_dim + reward_dim)
                obs = torch.cat((obs_obs, obs_act, obs_rew), dim=-1)

                obs_enc, s_n = super().forward(obs, s_0)  # obs_enc with shape (bs, logit_shape)
                obs_enc = obs_enc.reshape(-1, len(self.hidden_ind),
                                          self.num_colors)  # (bs, num_hidden_states, num_colors)
                obs_enc = F.gumbel_softmax(obs_enc, hard=True) if self.training \
                    else F.one_hot(torch.argmax(obs_enc, dim=-1), obs_enc.size(-1)).float()
            obs_enc = torch.unbind(obs_enc, dim=1)  # [(bs, (n_pred_step), num_colors)] * num_hidden_states

            """concatenate outputs of FNN and RNN"""
            num_hidden_objects = len(self.hidden_objects_ind)
            num_obs_objects = self.num_objects - len(self.hidden_objects_ind)
            if len(self.hidden_targets_ind) > 0:
                obs_enc = obs_obs_forward[:self.hidden_objects_ind[0]] + obs_enc[:num_hidden_objects] \
                          + obs_obs_forward[self.hidden_objects_ind[0]:(num_obs_objects + self.hidden_targets_ind[0])] \
                          + obs_enc[num_hidden_objects:] + obs_obs_forward[
                                                           (num_obs_objects + self.hidden_targets_ind[0]):]
            else:
                obs_enc = obs_obs_forward[:self.hidden_objects_ind[0]] + obs_enc[:num_hidden_objects] \
                          + obs_obs_forward[self.hidden_objects_ind[0]:]
            obs_enc = list(obs_enc)  # [(bs, (n_pred_step), num_colors)] * (2 * num_objects)
            return obs_enc[:self.num_objects], obs_enc[self.num_objects:]


class ForwardEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.device = params.device
        self.feedforward_enc_params = params.encoder_params.feedforward_enc_params

        self.continuous_state = params.continuous_state
        self.continuous_action = params.continuous_action

        chemical_env_params = params.env_params.chemical_env_params
        self.hidden_objects_ind = chemical_env_params.hidden_objects_ind
        self.hidden_targets_ind = chemical_env_params.hidden_targets_ind
        self.hidden_ind = params.hidden_ind
        self.num_objects = params.env_params.chemical_env_params.num_objects
        self.num_colors = params.env_params.chemical_env_params.num_colors
        self.num_hidden_objects = len(self.hidden_objects_ind)
        self.num_obs_objects = self.num_objects - len(self.hidden_objects_ind)

        self.gumbel_temp = self.feedforward_enc_params.gumbel_temp_0
        self.dropout = nn.Dropout(p=self.feedforward_enc_params.dropout)

        # d-set keys and all observable keys
        self.keys_dset = params.env_params.chemical_env_params.keys_dset_0
        self.keys_remapped_dset = self.keys_dset + ['act'] + ['rew']
        self.keys = [key for key in params.obs_keys if params.obs_spec[key].ndim == 1]

        # input features of encoder, where d-set features are fed to masked MLP
        self.stack_num = params.training_params.replay_buffer_params.stack_num  # t-1, t, t+1
        self.num_dset_obs = len(self.keys_dset)
        self.feature_dim_dset = (self.stack_num - 1) * self.num_dset_obs  # {feature_t^i, feature_{t+1}^i}
        self.continuous_state = params.continuous_state
        self.feature_inner_dim_obs, self.feature_inner_dim_dset_obs = None, None
        if not self.continuous_state:
            self.feature_inner_dim_obs = np.concatenate([params.obs_dims[key] for key in self.keys])
            self.feature_inner_dim_dset_obs = np.concatenate([params.obs_dims[key] for key in self.keys_dset])
            self.feature_inner_dim_dset = np.array([self.num_colors for _ in range(self.feature_dim_dset)])
            self.feature_inner_dim_hiddens = np.array([self.num_colors for _ in range(self.num_hidden_objects)])
        self.feature_inner_dim_dset_obses = np.append(self.feature_inner_dim_dset_obs, [params.action_dim, 1])

        # output features of encoder
        self.feature_dim = len(params.obs_keys_f)
        self.feature_inner_dim = None
        if not self.continuous_state:
            self.feature_inner_dim = np.concatenate([params.obs_dims_f[key] for key in params.obs_keys_f])

        self.hoa_counts = Counter()
        self.hoa_probs = {}
        self.init_model()
        self.reset_params()

    def init_model(self):
        params = self.params
        feedforward_enc_params = self.feedforward_enc_params

        # model params
        continuous_state = self.continuous_state

        self.action_dim = action_dim = params.action_dim
        feature_dim_dset = self.feature_dim_dset
        num_hidden_objects = self.num_hidden_objects

        # [(num_hidden_objects, in_dim_i, out_dim_i)] * len(feature_fc_dims)
        self.action_feature_weights = nn.ParameterList()
        # [(num_hidden_objects, 1, out_dim_i)] * len(feature_fc_dims)
        self.action_feature_biases = nn.ParameterList()

        # [(num_hidden_objects * feature_dim_dset, in_dim_i, out_dim_i)] * len(feature_fc_dims[1:]) for discrete state space
        self.state_feature_weights = nn.ParameterList()
        # [(num_hidden_objects * feature_dim_dset, 1, out_dim_i)] * len(feature_fc_dims[1:]) for discrete state space
        self.state_feature_biases = nn.ParameterList()

        # [(num_hidden_objects, in_dim_i, out_dim_i)] * len(generative_fc_dims)
        self.generative_weights = nn.ParameterList()
        # [(num_hidden_objects, 1, out_dim_i)] * len(generative_fc_dims)
        self.generative_biases = nn.ParameterList()

        # only needed for discrete state space
        # [(num_hidden_objects, feature_inner_dim_dset_i, out_dim)] * len(feature_inner_dim_dset)
        self.state_feature_1st_layer_weights = nn.ParameterList()
        # [(num_hidden_objects, 1, out_dim)] * len(feature_inner_dim_dset)
        self.state_feature_1st_layer_biases = nn.ParameterList()

        # [(1, in_dim, feature_inner_dim_hiddens_i)] * len(feature_inner_dim_hiddens)
        self.generative_last_layer_weights = nn.ParameterList()
        # [(1, 1, feature_inner_dim_hiddens_i)] * len(feature_inner_dim_hiddens)
        self.generative_last_layer_biases = nn.ParameterList()

        # Instantiate the parameters of each layer in the model of each variable at next time step to predict
        # action feature extractor
        in_dim = action_dim
        for out_dim in feedforward_enc_params.feature_fc_dims:
            self.action_feature_weights.append(nn.Parameter(torch.zeros(num_hidden_objects, in_dim, out_dim)))
            self.action_feature_biases.append(nn.Parameter(torch.zeros(num_hidden_objects, 1, out_dim)))
            in_dim = out_dim

        # state feature extractor
        if continuous_state:
            in_dim = 1
            fc_dims = feedforward_enc_params.feature_fc_dims
        else:
            out_dim = feedforward_enc_params.feature_fc_dims[0]
            fc_dims = feedforward_enc_params.feature_fc_dims[1:]
            for feature_i_dim_dset in self.feature_inner_dim_dset:
                in_dim = feature_i_dim_dset
                self.state_feature_1st_layer_weights.append(
                    nn.Parameter(torch.zeros(num_hidden_objects, in_dim, out_dim)))
                self.state_feature_1st_layer_biases.append(nn.Parameter(torch.zeros(num_hidden_objects, 1, out_dim)))
            in_dim = out_dim

        for out_dim in fc_dims:
            self.state_feature_weights.append(
                nn.Parameter(torch.zeros(num_hidden_objects * feature_dim_dset, in_dim, out_dim)))
            self.state_feature_biases.append(
                nn.Parameter(torch.zeros(num_hidden_objects * feature_dim_dset, 1, out_dim)))
            in_dim = out_dim

        # predictor
        in_dim = feedforward_enc_params.feature_fc_dims[-1]
        for out_dim in feedforward_enc_params.generative_fc_dims:
            self.generative_weights.append(nn.Parameter(torch.zeros(num_hidden_objects, in_dim, out_dim)))
            self.generative_biases.append(nn.Parameter(torch.zeros(num_hidden_objects, 1, out_dim)))
            in_dim = out_dim

        if continuous_state:
            self.generative_weights.append(nn.Parameter(torch.zeros(num_hidden_objects, in_dim, 2)))
            self.generative_biases.append(nn.Parameter(torch.zeros(num_hidden_objects, 1, 2)))
        else:
            for feature_i_dim_hiddens in self.feature_inner_dim_hiddens:
                final_dim = 2 if feature_i_dim_hiddens == 1 else feature_i_dim_hiddens
                self.generative_last_layer_weights.append(nn.Parameter(torch.zeros(1, in_dim, final_dim)))
                self.generative_last_layer_biases.append(nn.Parameter(torch.zeros(1, 1, final_dim)))

    def reset_params(self):
        for w, b in zip(self.action_feature_weights, self.action_feature_biases):
            for i in range(self.num_hidden_objects):
                reset_layer(w[i], b[i])
        for w, b in zip(self.state_feature_1st_layer_weights, self.state_feature_1st_layer_biases):
            for i in range(self.num_hidden_objects):
                reset_layer(w[i], b[i])
        for w, b in zip(self.state_feature_weights, self.state_feature_biases):
            for i in range(self.num_hidden_objects * self.feature_dim_dset):
                reset_layer(w[i], b[i])
        for w, b in zip(self.generative_weights, self.generative_biases):
            for i in range(self.num_hidden_objects):
                reset_layer(w[i], b[i])
        for w, b in zip(self.generative_last_layer_weights, self.generative_last_layer_biases):
            reset_layer(w, b)

    def extract_action_feature(self, action):
        """
        :param action: (bs, action_dim)
        :return: (num_hidden_objects, 1, bs, out_dim)
        """
        action = action.unsqueeze(dim=0)  # (1, bs, action_dim)
        action = action.expand(self.num_hidden_objects, -1, -1)  # (num_hidden_objects, bs, action_dim)
        action_feature = forward_network(action,
                                         self.action_feature_weights,
                                         self.action_feature_biases,
                                         dropout=self.dropout)  # (num_hidden_objects, bs, feature_fc_dims[-1])
        return action_feature.unsqueeze(dim=1)  # (num_hidden_objects, 1, bs, out_dim)

    def extract_state_feature(self, feature):
        """
        :param feature:
            if state space is continuous: (bs, feature_dim)
            else: [(bs, num_colors)] * feature_dim_dset
        :return: (num_hidden_objects, feature_dim_dset, bs, out_dim),
            the first dim is each state variable to predict, the second dim is inputs for the prediction
        """
        num_hidden_objects = self.num_hidden_objects
        feature_dim_dset = self.feature_dim_dset
        if self.continuous_state:
            bs = feature.shape[0]
            # x = feature.transpose(0, 1)  # (feature_dim, bs)
            # x = x.repeat(feature_dim, 1, 1)  # (feature_dim, feature_dim, bs)
            # x = x.view(feature_dim * feature_dim, bs, 1)  # (feature_dim * feature_dim, bs, 1)
        else:
            bs = feature[0].shape[0]
            # [(num_hidden_objects, bs, num_colors)] * feature_dim_dset
            reshaped_feature = []
            for f_i in feature:
                f_i = f_i.repeat(num_hidden_objects, 1, 1)  # (num_hidden_objects, bs, num_colors)
                reshaped_feature.append(f_i)
            x = forward_network_batch(reshaped_feature,
                                      self.state_feature_1st_layer_weights,
                                      self.state_feature_1st_layer_biases,
                                      dropout=self.dropout)  # [(num_hidden_objects, bs, out_dim)] * feature_dim_dset
            x = torch.stack(x, dim=1)  # (num_hidden_objects, feature_dim_dset, bs, out_dim)
            # (num_hidden_objects * feature_dim_dset, bs, out_dim)
            x = x.view(num_hidden_objects * feature_dim_dset, *x.shape[2:])

        # (num_hidden_objects * feature_dim_dset, bs, out_dim)
        state_feature = forward_network(x, self.state_feature_weights, self.state_feature_biases, dropout=self.dropout)
        state_feature = state_feature.view(num_hidden_objects, feature_dim_dset, bs, -1)
        return state_feature  # (num_hidden_objects, feature_dim_dset, bs, out_dim)

    def predict_from_sa_feature(self, sa_feature, residual_base=None):
        """
        predict the distribution and sample for the next step value of all state variables
        :param sa_feature: (num_hidden_objects, bs, out_dim)
        :param residual_base: (bs, feature_dim), residual used for continuous state variable prediction
        :return: if state space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: [OneHotCategorical / Normal] * num_hidden_objects, each of shape (bs, num_colors)
        """
        x = forward_network(sa_feature,
                            self.generative_weights,
                            self.generative_biases,
                            dropout=self.dropout)  # (num_hidden_objects, bs, out_dim)

        if self.continuous_state:
            x = x.permute(1, 0, 2)  # (bs, feature_dim, 2)
            mu, log_std = x.unbind(dim=-1)  # (bs, feature_dim) * 2
            return self.normal_helper(mu, residual_base, log_std)
        else:
            x = F.relu(x)  # (num_hidden_objects, bs, out_dim)
            x = [x_i.unsqueeze(dim=0) for x_i in torch.unbind(x, dim=0)]  # [(1, bs, out_dim)] * num_hidden_objects
            x = forward_network_batch(x,
                                      self.generative_last_layer_weights,
                                      self.generative_last_layer_biases,
                                      activation=None,
                                      dropout=self.dropout)  # [(1, bs, num_colors)] * num_hidden_objects

            dist = []
            for feature_i_inner_dim_hiddens, dist_i in zip(self.feature_inner_dim_hiddens, x):
                dist_i = dist_i.squeeze(dim=0)  # (bs, num_colors)
                if feature_i_inner_dim_hiddens == 1:
                    mu, log_std = torch.split(dist_i, 1, dim=-1)  # (bs, 1), (bs, 1)
                    # dist.append(self.normal_helper(mu, base_i, log_std))
                else:
                    dist.append(OneHotCategorical(logits=dist_i))
            return dist

    def forward_step(self, full_feature, masked_feature, action, mask=None):
        """
        :param full_feature: if state space is continuous: (bs, feature_dim).
            Otherwise: [(bs, num_colors)] * feature_dim_dset
            if it is None, no need to forward it
        :param masked_feature: (bs, feature_dim) or [(bs, num_colors)] * feature_dim_dset
        :param action: (bs, action_dim)
        :param mask: (bs, num_hidden_objects, feature_dim_dset + 1)
        :return: if state space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: [OneHotCategorical / Normal] * num_hidden_objects, each of shape (bs, num_colors)
        """
        forward_full = full_feature is not None
        forward_masked = masked_feature is not None
        full_state_feature = None
        full_dist = masked_dist = None

        # extract features of the action
        # (bs, action_dim) -> (num_hidden_objects, 1, bs, out_dim)
        self.action_feature = action_feature = self.extract_action_feature(action)

        if forward_full:
            # 1. extract features of all state variables
            # [(bs, num_colors)] * feature_dim_dset -> (num_hidden_objects, feature_dim_dset, bs, out_dim)
            self.full_state_feature = full_state_feature = self.extract_state_feature(full_feature)

            # 2. extract global feature by element-wise max
            # (num_hidden_objects, feature_dim_dset + 1, bs, out_dim)
            full_sa_feature = torch.cat([full_state_feature, action_feature], dim=1)
            full_sa_feature, full_sa_indices = full_sa_feature.max(dim=1)  # (num_hidden_objects, bs, out_dim)

            # 3. predict the distribution of next time step value
            full_dist = self.predict_from_sa_feature(full_sa_feature,
                                                     full_feature)  # [(bs, num_colors)] * num_hidden_objects

        if forward_masked:
            # 1. extract features of all state variables
            # (num_hidden_objects, feature_dim_dset, bs, out_dim)
            masked_state_feature = self.extract_state_feature(masked_feature)

            # 2. extract global feature by element-wise max
            # mask out unused features
            # (num_hidden_objects, feature_dim_dset + 1, bs, out_dim)
            masked_sa_feature = torch.cat([masked_state_feature, action_feature], dim=1)
            mask = mask.permute(1, 2, 0)  # (num_hidden_objects, feature_dim_dset + 1, bs)
            masked_sa_feature[~mask] = float('-inf')
            masked_sa_feature, masked_sa_indices = masked_sa_feature.max(dim=1)  # (num_hidden_objects, bs, out_dim)

            # 3. predict the distribution of next time step value
            masked_dist = self.predict_from_sa_feature(masked_sa_feature,
                                                       masked_feature)  # [(bs, num_colors)] * num_hidden_objects

        return full_dist, masked_dist

    def sample_from_distribution(self, dist):
        """
        sample from the distribution
        :param dist:
            if state space is continuous: Normal distribution of shape (bs, feature_dim).
            else: [OneHotCategorical / Normal] * num_hidden_objects, each of shape(bs, num_colors)
        :return:
            if state space is continuous: (bs, feature_dim)
            else: [(bs, num_colors)]  * num_hidden_objects
        """
        if self.continuous_state:
            return dist.rsample() if self.training else dist.mean
        else:
            sample = []
            for dist_i in dist:
                if isinstance(dist_i, Normal):
                    sample_i = dist_i.rsample() if self.training else dist_i.mean
                elif isinstance(dist_i, OneHotCategorical):
                    logits = dist_i.logits
                    if self.training:
                        sample_i = F.gumbel_softmax(logits, tau=self.gumbel_temp, hard=True)
                    else:
                        sample_i = F.one_hot(torch.argmax(logits, dim=-1), logits.size(-1)).float()
                else:
                    raise NotImplementedError
                sample.append(sample_i)
            return sample

    def stack_dist(self, dist_list):
        """
        list of distribution at different time steps to a single distribution stacked at dim=-2
        :param dist_list:
            if state space is continuous: [Normal] * n_pred_step, each of shape (bs, feature_dim)
            else: [[OneHotCategorical / Normal] * feature_dim] * n_pred_step, each of shape (bs, feature_i_dim)
        :return:
            if state space is continuous: Normal distribution of shape (bs, n_pred_step, feature_dim)
            else: [OneHotCategorical / Normal]  * feature_dim, each of shape (bs, n_pred_step, feature_i_dim)
        """
        if self.continuous_state:
            mu = torch.stack([dist.mean for dist in dist_list], dim=-2)  # (bs, n_pred_step, feature_dim)
            std = torch.stack([dist.stddev for dist in dist_list], dim=-2)  # (bs, n_pred_step, feature_dim)
            return Normal(mu, std)
        else:
            # [(bs, n_pred_step, feature_i_dim)]
            stacked_dist_list = []
            for i, dist_i in enumerate(dist_list[0]):
                if isinstance(dist_i, Normal):
                    # (bs, n_pred_step, feature_i_dim)
                    mu = torch.stack([dist[i].mean for dist in dist_list], dim=-2)
                    std = torch.stack([dist[i].stddev for dist in dist_list], dim=-2)
                    stacked_dist_i = Normal(mu, std)
                elif isinstance(dist_i, OneHotCategorical):
                    # (bs, n_pred_step, feature_i_dim)
                    logits = torch.stack([dist[i].logits for dist in dist_list], dim=-2)
                    stacked_dist_i = OneHotCategorical(logits=logits)
                elif isinstance(dist_i, torch.Tensor):
                    # (bs, n_pred_step, feature_i_dim)
                    stacked_dist_i = torch.stack([dist[i] for dist in dist_list], dim=-2)
                else:
                    raise NotImplementedError
                stacked_dist_list.append(stacked_dist_i)
            return stacked_dist_list

    def forward_with_feature(self, features, actions, mask=None, forward_mode=("full", "masked")):
        """
        :param features: (bs, feature_dim) if state space is continuous
            else [[(bs, num_colors)] * feature_dim_dset] * n_pred_step or [(bs, num_colors)] * feature_dim_dset
        :param actions: [(bs, action_dim)] * n_pred_step or (bs, action_dim)
        :param mask: (bs, num_hidden_objects, feature_dim_dset + 1)
        :param forward_mode
        :return result_dists: [[OneHotCategorical / Normal] * num_hidden_objects] * len(forward_mode),
            each of shape (bs, n_pred_step, num_colors) or (bs, num_colors)
        :return result_features: gumbel-softmax samples
        """

        if "masked" in forward_mode:
            assert mask is not None

        if isinstance(actions, list):
            full_features = features if "full" in forward_mode else [None for _ in range(len(features))]
            masked_features = features if "masked" in forward_mode else [None for _ in range(len(features))]
            full_dists, masked_dists = [], []
            full_sample_features, masked_sample_features = [], []
            for full_feature, masked_feature, action in zip(full_features, masked_features, actions):
                # [OneHotCategorical / Normal] * num_hidden_objects, each of shape(bs, num_colors)
                full_dist, masked_dist = \
                    self.forward_step(full_feature, masked_feature, action, mask)  # one-step forward prediction

                # sample feature from the distribution
                full_sample_feature = self.sample_from_distribution(full_dist) if full_dist is not None else None
                masked_sample_feature = self.sample_from_distribution(masked_dist) if masked_dist is not None else None

                # [[(bs, num_colors)] * num_hidden_objects] * n_pred_step
                full_dists.append(full_dist)
                masked_dists.append(masked_dist)
                full_sample_features.append(full_sample_feature)
                masked_sample_features.append(masked_sample_feature)

            dists = [full_dists, masked_dists]
            features = [full_sample_features, masked_sample_features]
            result_dists = []
            result_features = []
            modes = ["full", "masked"]
            for mode in forward_mode:
                dist, feature = dists[modes.index(mode)], features[modes.index(mode)]
                # [(bs, n_pred_step, num_colors)] * num_hidden_objects
                dist, feature = self.stack_dist(dist), self.stack_dist(feature)
                result_dists.append(dist)
                result_features.append(feature)
            if len(forward_mode) == 1:
                return result_dists[0], result_features[0]

            return result_dists, result_features
        else:
            full_features = features if "full" in forward_mode else None
            masked_features = features if "masked" in forward_mode else None
            # [OneHotCategorical / Normal] * num_hidden_objects, each of shape(bs, num_colors)
            full_dist, masked_dist = \
                self.forward_step(full_features, masked_features, actions, mask)

            # sample feature from the distribution
            full_sample_feature = self.sample_from_distribution(full_dist) if full_dist is not None else None
            masked_sample_feature = self.sample_from_distribution(masked_dist) if masked_dist is not None else None

            dists = [full_dist, masked_dist]
            features = [full_sample_feature, masked_sample_feature]
            result_dists = []
            result_features = []
            modes = ["full", "masked"]
            for mode in forward_mode:
                # [(bs, num_colors)] * num_hidden_objects
                dist, feature = dists[modes.index(mode)], features[modes.index(mode)]
                result_dists.append(dist)
                result_features.append(feature)
            if len(forward_mode) == 1:
                return result_dists[0], result_features[0]

            return result_dists, result_features

    def get_mask(self, batch_size, i):
        # omit i-th state variable or the action when predicting the next time step value
        # by setting it to false
        # (bs, num_hidden_objects)
        idxes = torch.full((batch_size, self.num_hidden_objects), i, dtype=torch.int64)
        int_mask = F.one_hot(idxes, self.feature_dim_dset + 1)
        bool_mask = int_mask < 1
        return bool_mask  # (bs, num_hidden_objects, feature_dim_dset + 1)

    def preprocess_obs(self, obs):
        """
        :param obs: Batch(obs_i_key: (bs, stack_num, (n_pred_step), obs_i_shape))
        :return feature: [[(bs, num_colors)] * (num_dset_observables * (stack_num - 1))] * n_pred_step or
                          [(bs, num_colors)] * (num_dset_observables * (stack_num - 1))
                action: [(bs, action_dim)] * n_pred_step or
                         (bs, action_dim)
        """
        obs = [obs_k_i
               for k in self.keys_remapped_dset
               for obs_k_i in torch.unbind(obs[k], dim=-1)]
        obs = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
               for obs_i, obs_i_dim in zip(obs, self.feature_inner_dim_dset_obses)]

        # slice in object observables from observation
        # (num_dset_observables, bs, stack_num - 1, (n_pred_step), num_colors)
        obs_obs = torch.stack(obs[:-2], dim=0)[:, :, 1:]

        # slice in action and reward from observation
        # (bs, 1, (n_pred_step), action_dim / reward_dim)
        obs_act, obs_rew = obs[-2].clone()[:, 1:2], obs[-1].clone()[:, 1:2]

        if len(obs_obs.shape) == 5:
            # (bs, n_pred_step, num_colors, stack_num - 1, num_dset_observables)
            obs_obs = obs_obs.permute(1, 3, 4, 2, 0)
            # (bs, n_pred_step, num_colors, num_dset_observables * (stack_num - 1))
            obs_obs = obs_obs.reshape(obs_obs.shape[:3] + (-1,))

            # [[(bs, num_colors)] * (num_dset_observables * (stack_num - 1))] * n_pred_step
            feature = [torch.unbind(obs_obs_i, dim=-1)
                       for obs_obs_i in
                       torch.unbind(obs_obs, dim=1)]
            # [(bs, action_dim)] * n_pred_step
            action = [obs_act_i.squeeze(dim=1)
                      for obs_act_i in
                      torch.unbind(obs_act, dim=-2)]
        else:
            # (bs, num_colors, stack_num - 1, num_dset_observables)
            obs_obs = obs_obs.permute(1, 3, 2, 0)
            # (bs, num_colors, num_dset_observables * (stack_num - 1))
            obs_obs = obs_obs.reshape(obs_obs.shape[:2] + (-1,))

            # [(bs, num_colors)] * (num_dset_observables * (stack_num - 1))
            feature = torch.unbind(obs_obs, dim=-1)
            # (bs, action_dim)
            action = obs_act.squeeze(dim=1)
        return feature, action

    def feedforward_net(self, obs):
        """
        FNN as identity map of observables
        :param obs: Batch(obs_i_key: (bs, stack_num, (n_pred_step), obs_i_shape))
        :return obs_obs_forward: [(bs, (n_pred_step), num_colors)] * num_observables
        """
        obs_forward = [obs_k_i
                       for k in self.keys
                       for obs_k_i in torch.unbind(obs[k], dim=-1)]
        obs_forward = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
                       for obs_i, obs_i_dim in zip(obs_forward, self.feature_inner_dim_obs)]

        # (num_observables, bs, stack_num, (n_pred_step), num_colors)
        obs_obs_forward = torch.stack(obs_forward, dim=0)

        # slice the current obs from the stack_num history
        # [(bs, (n_pred_step), num_colors)] * num_observables
        obs_obs_forward = list(torch.unbind(obs_obs_forward[:, :, -2]))
        return obs_obs_forward

    def forward(self, obs, feature, action, mask=None, forward_mode=("full", "masked")):
        """
        masked MLP fed by d-set
        :param obs: Batch(obs_i_key: (bs, stack_num, (n_pred_step), obs_i_shape))
        :param feature: [[(bs, num_colors)] * (num_dset_observables * (stack_num - 1))] * n_pred_step or
                         [(bs, num_colors)] * (num_dset_observables * (stack_num - 1))
        :param action: [(bs, action_dim)] * n_pred_step or (bs, action_dim)
        :param mask: (bs, num_hidden_objects, feature_dim_dset + 1)
        the second dim represents binary d-sets at t, t+1 and binary action at t
        :param forward_mode
        :return enc_dist/enc_feature: [(bs, (n_pred_step), num_colors)] * num_hidden_objects
                hoa_llh: (bs, ) likelihoods of (hidden, obs, action)
        """
        bs = len(obs[next(iter(dict(obs)))])
        oa_batch = obs_batch_dict2tuple(obs, self.params)[0]

        # enc_dist/enc_feature: [(bs, (n_pred_step), num_colors)] * num_hidden_objects
        enc_dist, enc_feature = self.forward_with_feature(feature, action, mask, forward_mode=forward_mode)

        # Batch-wise update the counting-based probability P1 of each tuple (hidden, obs, action)
        if self.params.training_params.count_env_transition:
            hiddens_batch = [tuple(hidden.argmax(-1).squeeze().item() for hidden in hiddens_batch)
                             for hiddens_batch in zip(*enc_feature)]
            hoa_batch = [(hidden,) + oa for hidden, oa in zip(hiddens_batch, oa_batch)]
            self.hoa_counts.update(hoa_batch)
            self.hoa_probs = count2prob(self.hoa_counts)
            hoa_llh = count2prob_llh(self.hoa_probs, hoa_batch).to(self.device)
        else:
            hoa_llh = None
        return enc_dist, enc_feature, hoa_llh

    def concat_oh(self, obs_obs_forward, enc_feature=None):
        """
        concatenate outputs of FNN and masked MLP
        :param obs_obs_forward: [(bs, (n_pred_step), num_colors)] * num_observables
        :param enc_feature: [(bs, (n_pred_step), num_colors)] * num_hidden_objects
        :return enc_obs_obj/enc_obs_target: [(bs, (n_pred_step), num_colors)] * num_objects
        """
        # identity encoder in the fully observable
        if len(self.hidden_ind) == 0:
            # enc_obs_obj/enc_obs_target: [(bs, (n_pred_step), num_colors)] * num_objects
            enc_obs_obj, enc_obs_target = obs_obs_forward[: self.num_objects], obs_obs_forward[self.num_objects:]
            return enc_obs_obj, enc_obs_target

        enc_obs = None
        if len(self.hidden_targets_ind) > 0:
            enc_obs = (obs_obs_forward[: self.hidden_objects_ind[0]]
                       + enc_feature[: self.num_hidden_objects]
                       + obs_obs_forward[self.hidden_objects_ind[0]:
                                         (self.num_obs_objects + self.hidden_targets_ind[0])]
                       + enc_feature[self.num_hidden_objects:]
                       + obs_obs_forward[(self.num_obs_objects + self.hidden_targets_ind[0]):])
        else:
            if len(self.hidden_objects_ind) == 1:  # 1 hidden object
                enc_obs = (obs_obs_forward[: self.hidden_objects_ind[0]]
                           + enc_feature[: self.num_hidden_objects]
                           + obs_obs_forward[self.hidden_objects_ind[0]:])
            elif len(self.hidden_objects_ind) == 2:  # 2 hidden objects
                enc_obs = (obs_obs_forward[: self.hidden_objects_ind[0]]
                           + enc_feature[: 1]
                           + obs_obs_forward[self.hidden_objects_ind[0]: self.hidden_objects_ind[1] - 1]
                           + enc_feature[1:]
                           + obs_obs_forward[self.hidden_objects_ind[1] - 1:])
        # enc_obs_obj/enc_obs_target: [(bs, (n_pred_step), num_colors)] * num_objects
        enc_obs_obj, enc_obs_target = enc_obs[: self.num_objects], enc_obs[self.num_objects:]
        return enc_obs_obj, enc_obs_target


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.device = params.device
        self.encoder_params = params.encoder_params
        self.gumbel_temp = self.encoder_params.gumbel_softmax_params.gumbel_temp_0
        self.eps_encoder = self.encoder_params.eps_encoder
        self.count_transitions = self.encoder_params.count_transitions

        chemical_env_params = params.env_params.chemical_env_params
        self.hidden_objects_ind = chemical_env_params.hidden_objects_ind
        self.hidden_targets_ind = chemical_env_params.hidden_targets_ind
        self.hidden_ind = params.hidden_ind
        self.num_objects = chemical_env_params.num_objects
        self.num_colors = chemical_env_params.num_colors
        self.num_hidden_objects = len(self.hidden_objects_ind)
        self.num_obs_objects = self.num_objects - len(self.hidden_objects_ind)
        self.eps_dim = len(chemical_env_params.eps_p)

        self.o_keys = params.obs_keys[:self.num_obs_objects]
        self.o_inner_dim = np.concatenate([params.obs_dims[key] for key in self.o_keys])
        self.ot_keys = params.obs_keys
        self.ot_inner_dim = np.concatenate([params.obs_dims[key] for key in self.ot_keys])
        self.a_key, self.r_key = 'act', 'rew'
        self.a_inner_dim = params.action_dim
        self.feature_in_dim = 2 * self.num_objects if self.eps_encoder else self.num_objects
        self.feature_out_dim = self.num_objects
        feature_inner_dim = np.concatenate([params.obs_dims_f[key] for key in params.obs_keys_f])
        self.feature_in_inner_dim = np.concatenate([feature_inner_dim, np.full(self.num_objects, self.eps_dim)]) \
            if self.eps_encoder else feature_inner_dim
        self.feature_out_inner_dim = feature_inner_dim

        # hindsight-based encoder implemented by masked MLP
        self.feedforward_enc_params = params.encoder_params.feedforward_enc_params
        self.hind_feature_dim = 2 * self.num_obs_objects
        self.continuous_state = params.continuous_state
        self.hind_feature_inner_dim, self.hidden_feature_inner_dim = None, None
        if not self.continuous_state:
            self.hind_feature_inner_dim = np.array([self.num_colors for _ in range(self.hind_feature_dim)])
            self.hidden_feature_inner_dim = np.array([self.num_colors for _ in range(self.num_hidden_objects)])
        self.dropout = nn.Dropout(p=self.feedforward_enc_params.dropout_p)

        self.use_all_past = self.encoder_params.use_all_past
        self.use_all_future = self.encoder_params.use_all_future
        self.num_future = self.encoder_params.num_future
        self.num_past = self.encoder_params.num_past

        self.xuz_counts = Counter()
        self.xuz_probs = {}
        self.update_num = 0

        self.init_model()
        self.reset_params()
        self.to(self.device)

    def init_model(self):
        self.seq_len = self.params.training_params.replay_buffer_params.stack_num
        self.z_dim = z_dim = self.num_hidden_objects * self.num_colors
        self.xu_dim = xu_dim = self.o_inner_dim.sum() + self.a_inner_dim
        # self.xu_dim = xu_dim = self.o_inner_dim[:2].sum() + 2
        self.dim_rnn_g = dim_rnn_g = self.encoder_params.dim_rnn_g
        self.num_rnn_g = num_rnn_g = self.encoder_params.num_rnn_g
        self.zxu_dim = zxu_dim = self.z_dim + self.o_inner_dim.sum() + self.a_inner_dim
        self.x_dim = x_dim = self.o_inner_dim.sum()
        # self.zxu_dim = zxu_dim = 2 * self.num_colors + self.a_inner_dim
        self.xxu_dim = xxu_dim = 2 * self.o_inner_dim.sum() + self.a_inner_dim
        self.x3u2_dim = x3u2_dim = 3 * self.o_inner_dim.sum() + 2 * self.a_inner_dim
        self.xxui_dim = xxui_dim = 2 * self.o_inner_dim.sum() + self.a_inner_dim + self.seq_len
        self.xzuxz_dim = xzuxz_dim = self.o_inner_dim.sum() + self.z_dim + self.a_inner_dim + self.num_colors
        self.hh_dim = hh_dim = 2 * self.num_colors
        self.xz_dim = xz_dim = 2 * (self.o_inner_dim.sum() + self.num_colors)
        self.dims_mlp_m = dims_mlp_m = self.encoder_params.dims_mlp_m
        self.dims_cf_n = dims_cf_n = self.encoder_params.dims_cf_n
        self.xxuzz_dim = xxuzz_dim = 2 * self.o_inner_dim.sum() + self.a_inner_dim + 2 * self.z_dim
        dropout_p = self.encoder_params.dropout_p

        # x_{t:T}, u_{t:T} -> g_t
        # self.rnn_g = nn.LSTM(xxu_dim, dim_rnn_g, num_rnn_g, batch_first=True)
        # self.rnn_g = nn.LSTM(xxuz_dim, dim_rnn_g, num_rnn_g, batch_first=True)
        # self.rnn_g = nn.LSTM(xu_dim, dim_rnn_g, num_rnn_g, batch_first=True)

        # independent RNN for each hidden object
        self.rnn_g = nn.ModuleList()
        for i in range(self.num_hidden_objects):
            self.rnn_g.append(nn.LSTM(xu_dim if self.use_all_future or self.use_all_past else xxu_dim,
                                      dim_rnn_g, num_rnn_g, batch_first=True))
            # self.rnn_g.append(nn.LSTM(xu_dim if self.use_all_future or self.use_all_past else x3u2_dim,
            #                           dim_rnn_g, num_rnn_g, batch_first=True))

        # independent RNN for each noise
        self.rnn_e = nn.ModuleList()
        for i in range(self.num_objects):
            self.rnn_e.append(nn.LSTM(xzuxz_dim, dim_rnn_g, num_rnn_g, batch_first=True))

        # # independent MLP for each hidden object and time step
        # self.rnn_g = nn.ModuleList()
        # for i in range(self.seq_len):
        #     rnn_g_h = nn.ModuleList()
        #     for j in range(self.num_hidden_objects):
        #         rnn_g_h.append(nn.LSTM(xxu_dim, dim_rnn_g, num_rnn_g, batch_first=True))
        #     self.rnn_g.append(rnn_g_h)

        # z_{t-1}, x_{t-1:t:2}, u_{t-1} -> m_t
        self.mlp_m = nn.ModuleList()
        for i in range(self.num_hidden_objects):
            dic_layers = OrderedDict()
            for n in range(len(dims_mlp_m)):
                if n == 0:
                    # dic_layers['linear' + str(n)] = nn.Linear(zxu_dim, dims_mlp_m[n])
                    # dic_layers['linear' + str(n)] = nn.Linear(xu_dim, dims_mlp_m[n])
                    dic_layers['linear' + str(n)] = nn.Linear(xxu_dim, dims_mlp_m[n])
                    # dic_layers['linear' + str(n)] = nn.Linear(x_dim, dims_mlp_m[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(dims_mlp_m[n - 1], dims_mlp_m[n])
                dic_layers['layer_norm' + str(n)] = nn.LayerNorm(dims_mlp_m[n])
                dic_layers['activation' + str(n)] = nn.ReLU()
                dic_layers['dropout' + str(n)] = nn.Dropout(p=dropout_p)
            dic_layers['linear_last'] = nn.Linear(dims_mlp_m[-1], dim_rnn_g)
            # dic_layers['linear_last'] = nn.Linear(x_dim, dim_rnn_g)
            dic_layers['layer_norm_last'] = nn.LayerNorm(dim_rnn_g)
            dic_layers['activation_last'] = nn.ReLU()
            dic_layers['dropout_last'] = nn.Dropout(p=dropout_p)
            self.mlp_m.append(nn.Sequential(dic_layers))

        # # hindsight-based MLP encoder for first connected hiddens
        # dic_layers = OrderedDict()
        # for n in range(len(dims_mlp_m)):
        #     if n == 0:
        #         dic_layers['linear' + str(n)] = nn.Linear(xxu_dim, dims_mlp_m[n])
        #     else:
        #         dic_layers['linear' + str(n)] = nn.Linear(dims_mlp_m[n - 1], dims_mlp_m[n])
        #     dic_layers['layer_norm' + str(n)] = nn.LayerNorm(dims_mlp_m[n])
        #     dic_layers['activation' + str(n)] = nn.ReLU()
        #     dic_layers['dropout' + str(n)] = nn.Dropout(p=dropout_p)
        # dic_layers['linear_last'] = nn.Linear(dims_mlp_m[-1], self.num_colors)
        # self.mlp_m = nn.Sequential(dic_layers)
        #
        # # hindsight-based MLP encoder for second connected hiddens
        # dic_layers = OrderedDict()
        # for n in range(len(dims_mlp_m)):
        #     if n == 0:
        #         dic_layers['linear' + str(n)] = nn.Linear(xz_dim, dims_mlp_m[n])
        #     else:
        #         dic_layers['linear' + str(n)] = nn.Linear(dims_mlp_m[n - 1], dims_mlp_m[n])
        #     dic_layers['layer_norm' + str(n)] = nn.LayerNorm(dims_mlp_m[n])
        #     dic_layers['activation' + str(n)] = nn.ReLU()
        #     dic_layers['dropout' + str(n)] = nn.Dropout(p=dropout_p)
        # dic_layers['linear_last'] = nn.Linear(dims_mlp_m[-1], self.num_colors)
        # self.mlp_m1 = nn.Sequential(dic_layers)

        # # m_t, g_t -> n_t; shared MLP by hidden objects
        # dic_layers = OrderedDict()
        # for n in range(len(dims_cf_n)):
        #     if n == 0:
        #         dic_layers['linear' + str(n)] = nn.Linear(dim_rnn_g, dims_cf_n[n])
        #         # dic_layers['linear' + str(n)] = nn.Linear(2 * dim_rnn_g, dims_cf_n[n])
        #     else:
        #         dic_layers['linear' + str(n)] = nn.Linear(dims_cf_n[n - 1], dims_cf_n[n])
        #     dic_layers['layer_norm' + str(n)] = nn.LayerNorm(dims_cf_n[n])
        #     dic_layers['activation' + str(n)] = nn.ReLU()
        #     dic_layers['dropout' + str(n)] = nn.Dropout(p=dropout_p)
        # dic_layers['linear_last'] = nn.Linear(dims_cf_n[-1], z_dim)
        # self.cf_n = nn.Sequential(dic_layers)

        # m_t, g_t -> n_t; independent MLP for each hidden object
        self.cf_n = nn.ModuleList()
        for i in range(2):
            cf_n_h = nn.ModuleList()
            for j in range(self.num_hidden_objects):
                dic_layers = OrderedDict()
                for n in range(len(dims_cf_n)):
                    if n == 0:
                        # dic_layers['linear' + str(n)] = nn.Linear(dim_rnn_g if i == 0 else 2 * dim_rnn_g,
                        #                                           dims_cf_n[n])
                        # dic_layers['linear' + str(n)] = nn.Linear(
                        #     2 * dim_rnn_g + self.seq_len - self.num_hidden_objects, dims_cf_n[n]
                        # )
                        if (self.use_all_past and self.use_all_future) or self.num_future > 0:
                            dim_cf_in = 2 * dim_rnn_g
                        else:
                            dim_cf_in = dim_rnn_g
                        dic_layers['linear' + str(n)] = nn.Linear(dim_cf_in, dims_cf_n[n])
                    else:
                        dic_layers['linear' + str(n)] = nn.Linear(dims_cf_n[n - 1], dims_cf_n[n])
                    dic_layers['layer_norm' + str(n)] = nn.LayerNorm(dims_cf_n[n])
                    dic_layers['activation' + str(n)] = nn.ReLU()
                    dic_layers['dropout' + str(n)] = nn.Dropout(p=dropout_p)
                dic_layers['linear_last'] = nn.Linear(dims_cf_n[-1], self.num_colors)
                cf_n_h.append(nn.Sequential(dic_layers))
            self.cf_n.append(cf_n_h)

        # independent MLP for each noise following RNN
        self.mlp_e = nn.ModuleList()
        for j in range(self.num_objects):
            dic_layers = OrderedDict()
            for n in range(len(dims_cf_n)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(dim_rnn_g, dims_cf_n[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(dims_cf_n[n - 1], dims_cf_n[n])
                dic_layers['layer_norm' + str(n)] = nn.LayerNorm(dims_cf_n[n])
                dic_layers['activation' + str(n)] = nn.ReLU()
                dic_layers['dropout' + str(n)] = nn.Dropout(p=dropout_p)
            dic_layers['linear_last'] = nn.Linear(dims_cf_n[-1], self.eps_dim)
            self.mlp_e.append(nn.Sequential(dic_layers))

        params = self.params
        feedforward_enc_params = self.feedforward_enc_params
        continuous_state = self.continuous_state
        self.action_dim = action_dim = params.action_dim
        hind_feature_dim = self.hind_feature_dim
        num_hidden_objects = self.num_hidden_objects

        # [(num_hidden_objects, in_dim_i, out_dim_i)] * len(feature_fc_dims)
        self.action_feature_weights = nn.ParameterList()
        # [(num_hidden_objects, 1, out_dim_i)] * len(feature_fc_dims)
        self.action_feature_biases = nn.ParameterList()

        # [(num_hidden_objects * hind_feature_dim, in_dim_i, out_dim_i)] * len(feature_fc_dims[1:]) for discrete state
        self.state_feature_weights = nn.ParameterList()
        # [(num_hidden_objects * hind_feature_dim, 1, out_dim_i)] * len(feature_fc_dims[1:]) for discrete state
        self.state_feature_biases = nn.ParameterList()

        # [(num_hidden_objects, in_dim_i, out_dim_i)] * len(generative_fc_dims)
        self.generative_weights = nn.ParameterList()
        # [(num_hidden_objects, 1, out_dim_i)] * len(generative_fc_dims)
        self.generative_biases = nn.ParameterList()

        # only needed for discrete state space
        # [(num_hidden_objects, hind_feature_inner_dim_i, out_dim)] * len(hind_feature_inner_dim)
        self.state_feature_1st_layer_weights = nn.ParameterList()
        # [(num_hidden_objects, 1, out_dim)] * len(hind_feature_inner_dim)
        self.state_feature_1st_layer_biases = nn.ParameterList()

        # [(1, in_dim, hidden_feature_inner_dim_i)] * len(hidden_feature_inner_dim)
        self.generative_last_layer_weights = nn.ParameterList()
        # [(1, 1, hidden_feature_inner_dim_i)] * len(hidden_feature_inner_dim)
        self.generative_last_layer_biases = nn.ParameterList()

        # Instantiate the parameters of each layer in the model of each variable at next time step to predict
        # action feature extractor
        in_dim = action_dim
        for out_dim in feedforward_enc_params.feature_fc_dims:
            self.action_feature_weights.append(
                nn.Parameter(torch.zeros(num_hidden_objects, in_dim, out_dim)))
            self.action_feature_biases.append(
                nn.Parameter(torch.zeros(num_hidden_objects, 1, out_dim)))
            in_dim = out_dim

        # state feature extractor
        if continuous_state:
            in_dim = 1
            fc_dims = feedforward_enc_params.feature_fc_dims
        else:
            out_dim = feedforward_enc_params.feature_fc_dims[0]
            fc_dims = feedforward_enc_params.feature_fc_dims[1:]
            for feature_i_dim_dset in self.hind_feature_inner_dim:
                in_dim = feature_i_dim_dset
                self.state_feature_1st_layer_weights.append(
                    nn.Parameter(torch.zeros(num_hidden_objects, in_dim, out_dim)))
                self.state_feature_1st_layer_biases.append(
                    nn.Parameter(torch.zeros(num_hidden_objects, 1, out_dim)))
            in_dim = out_dim

        for out_dim in fc_dims:
            self.state_feature_weights.append(
                nn.Parameter(torch.zeros(num_hidden_objects * hind_feature_dim, in_dim, out_dim)))
            self.state_feature_biases.append(
                nn.Parameter(torch.zeros(num_hidden_objects * hind_feature_dim, 1, out_dim)))
            in_dim = out_dim

        # predictor
        in_dim = feedforward_enc_params.feature_fc_dims[-1]
        for out_dim in feedforward_enc_params.generative_fc_dims:
            self.generative_weights.append(
                nn.Parameter(torch.zeros(num_hidden_objects, in_dim, out_dim)))
            self.generative_biases.append(
                nn.Parameter(torch.zeros(num_hidden_objects, 1, out_dim)))
            in_dim = out_dim

        if continuous_state:
            self.generative_weights.append(
                nn.Parameter(torch.zeros(num_hidden_objects, in_dim, 2)))
            self.generative_biases.append(
                nn.Parameter(torch.zeros(num_hidden_objects, 1, 2)))
        else:
            for feature_i_dim_hiddens in self.hidden_feature_inner_dim:
                final_dim = 2 if feature_i_dim_hiddens == 1 else feature_i_dim_hiddens
                self.generative_last_layer_weights.append(
                    nn.Parameter(torch.zeros(1, in_dim, final_dim)))
                self.generative_last_layer_biases.append(
                    nn.Parameter(torch.zeros(1, 1, final_dim)))

    def reset_params(self):
        for w, b in zip(self.action_feature_weights, self.action_feature_biases):
            for i in range(self.num_hidden_objects):
                reset_layer(w[i], b[i])
        for w, b in zip(self.state_feature_1st_layer_weights, self.state_feature_1st_layer_biases):
            for i in range(self.num_hidden_objects):
                reset_layer(w[i], b[i])
        for w, b in zip(self.state_feature_weights, self.state_feature_biases):
            for i in range(self.num_hidden_objects * self.hind_feature_dim):
                reset_layer(w[i], b[i])
        for w, b in zip(self.generative_weights, self.generative_biases):
            for i in range(self.num_hidden_objects):
                reset_layer(w[i], b[i])
        for w, b in zip(self.generative_last_layer_weights, self.generative_last_layer_biases):
            reset_layer(w, b)

    def extract_action_feature(self, action):
        """
        :param action: (bs, action_dim)
        :return: (num_hidden_objects, 1, bs, out_dim)
        """
        action = action.unsqueeze(dim=0)  # (1, bs, action_dim)
        action = action.expand(self.num_hidden_objects, -1, -1)  # (num_hidden_objects, bs, action_dim)
        action_feature = forward_network(action,
                                         self.action_feature_weights,
                                         self.action_feature_biases,
                                         dropout=self.dropout)  # (num_hidden_objects, bs, feature_fc_dims[-1])
        return action_feature.unsqueeze(dim=1)  # (num_hidden_objects, 1, bs, out_dim)

    def extract_state_feature(self, feature):
        """
        :param feature:
            if state space is continuous: (bs, feature_dim)
            else: [(bs, num_colors)] * hind_feature_dim
        :return: (num_hidden_objects, hind_feature_dim, bs, out_dim),
            the first dim is each state variable to predict, the second dim is inputs for the prediction
        """
        num_hidden_objects = self.num_hidden_objects
        hind_feature_dim = self.hind_feature_dim
        if self.continuous_state:
            bs = feature.shape[0]
            # x = feature.transpose(0, 1)  # (feature_dim, bs)
            # x = x.repeat(feature_dim, 1, 1)  # (feature_dim, feature_dim, bs)
            # x = x.view(feature_dim * feature_dim, bs, 1)  # (feature_dim * feature_dim, bs, 1)
        else:
            bs = feature[0].shape[0]
            # [(num_hidden_objects, bs, num_colors)] * hind_feature_dim
            reshaped_feature = []
            for f_i in feature:
                f_i = f_i.repeat(num_hidden_objects, 1, 1)  # (num_hidden_objects, bs, num_colors)
                reshaped_feature.append(f_i)
            # [(num_hidden_objects, bs, out_dim)] * hind_feature_dim
            x = forward_network_batch(reshaped_feature,
                                      self.state_feature_1st_layer_weights,
                                      self.state_feature_1st_layer_biases,
                                      dropout=self.dropout)
            x = torch.stack(x, dim=1)  # (num_hidden_objects, hind_feature_dim, bs, out_dim)
            # (num_hidden_objects * hind_feature_dim, bs, out_dim)
            x = x.view(num_hidden_objects * hind_feature_dim, *x.shape[2:])

        # (num_hidden_objects * hind_feature_dim, bs, out_dim)
        state_feature = forward_network(x,
                                        self.state_feature_weights,
                                        self.state_feature_biases,
                                        dropout=self.dropout)
        state_feature = state_feature.view(num_hidden_objects, hind_feature_dim, bs, -1)
        return state_feature  # (num_hidden_objects, hind_feature_dim, bs, out_dim)

    def predict_from_sa_feature(self, sa_feature, residual_base=None):
        """
        predict the distribution for the next step value of all state variables
        :param sa_feature: (num_hidden_objects, bs, out_dim)
        :param residual_base: (bs, feature_dim), residual used for continuous state variable prediction
        :return: if state space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: [(bs, num_colors)] * num_hidden_objects
        """
        x = forward_network(sa_feature,
                            self.generative_weights,
                            self.generative_biases,
                            dropout=self.dropout)  # (num_hidden_objects, bs, out_dim)

        if self.continuous_state:
            x = x.permute(1, 0, 2)  # (bs, feature_dim, 2)
            mu, log_std = x.unbind(dim=-1)  # (bs, feature_dim) * 2
            return self.normal_helper(mu, residual_base, log_std)
        else:
            x = F.relu(x)  # (num_hidden_objects, bs, out_dim)
            x = [x_i.unsqueeze(dim=0) for x_i in torch.unbind(x, dim=0)]  # [(1, bs, out_dim)] * num_hidden_objects
            x = forward_network_batch(x,
                                      self.generative_last_layer_weights,
                                      self.generative_last_layer_biases,
                                      activation=None,
                                      dropout=self.dropout)  # [(1, bs, num_colors)] * num_hidden_objects

            dist = []
            for feature_i_inner_dim_hiddens, dist_i in zip(self.hidden_feature_inner_dim, x):
                dist_i = dist_i.squeeze(dim=0)  # (bs, num_colors)
                if feature_i_inner_dim_hiddens == 1:
                    mu, log_std = torch.split(dist_i, 1, dim=-1)  # (bs, 1), (bs, 1)
                    # dist.append(self.normal_helper(mu, base_i, log_std))
                else:
                    dist.append(dist_i)
            return dist

    def forward_step(self, full_feature, masked_feature, action, mask=None):
        """
        :param full_feature: if state space is continuous: (bs, feature_dim).
            Otherwise: [(bs, num_colors)] * hind_feature_dim
            if it is None, no need to forward it
        :param masked_feature: (bs, feature_dim) or [(bs, num_colors)] * hind_feature_dim
        :param action: (bs, action_dim)
        :param mask: (bs, num_hidden_objects, hind_feature_dim + 1)
        :return: if state space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: [OneHotCategorical / Normal] * num_hidden_objects, each of shape (bs, num_colors)
        """
        forward_full = full_feature is not None
        forward_masked = masked_feature is not None
        full_state_feature = None
        full_dist = masked_dist = None

        # extract features of the action
        # (bs, action_dim) -> (num_hidden_objects, 1, bs, out_dim)
        self.action_feature = action_feature = self.extract_action_feature(action)

        if forward_full:
            # 1. extract features of all state variables
            # [(bs, num_colors)] * hind_feature_dim -> (num_hidden_objects, hind_feature_dim, bs, out_dim)
            self.full_state_feature = full_state_feature = self.extract_state_feature(full_feature)

            # 2. extract global feature by element-wise max
            # (num_hidden_objects, hind_feature_dim + 1, bs, out_dim)
            full_sa_feature = torch.cat([full_state_feature, action_feature], dim=1)
            full_sa_feature, full_sa_indices = full_sa_feature.max(dim=1)           # (num_hidden_objects, bs, out_dim)

            # 3. predict the distribution of next time step value
            full_dist = self.predict_from_sa_feature(full_sa_feature, full_feature)  # [(bs, num_colors)] * num_hidden_objects

        if forward_masked:
            # 1. extract features of all state variables
            # (num_hidden_objects, hind_feature_dim, bs, out_dim)
            masked_state_feature = self.extract_state_feature(masked_feature)

            # 2. extract global feature by element-wise max
            # mask out unused features
            # (num_hidden_objects, hind_feature_dim + 1, bs, out_dim)
            masked_sa_feature = torch.cat([masked_state_feature, action_feature], dim=1)
            mask = mask.permute(1, 2, 0)                                            # (num_hidden_objects, hind_feature_dim + 1, bs)
            masked_sa_feature[~mask] = float('-inf')
            masked_sa_feature, masked_sa_indices = masked_sa_feature.max(dim=1)     # (num_hidden_objects, bs, out_dim)

            # 3. predict the distribution of next time step value
            masked_dist = self.predict_from_sa_feature(masked_sa_feature, masked_feature)  # [(bs, num_colors)] * num_hidden_objects

        return full_dist, masked_dist

    def forward_with_feature(self, features, actions, mask=None, forward_mode=("full", "masked")):
        """
        :param features: (bs, feature_dim) if state space is continuous
            else [(bs, num_colors)] * hind_feature_dim
        :param actions: (bs, action_dim)
        :param mask: (bs, num_hidden_objects, hind_feature_dim + 1)
        :param forward_mode
        :return result_dists: [[(bs, num_colors)] * num_hidden_objects] * len(forward_mode)
        """

        if "masked" in forward_mode:
            assert mask is not None

        full_features = features if "full" in forward_mode else None
        masked_features = features if "masked" in forward_mode else None
        # [(bs, num_colors)] * num_hidden_objects
        full_dist, masked_dist = \
            self.forward_step(full_features, masked_features, actions, mask)

        dists = [full_dist, masked_dist]
        result_dists = []
        modes = ["full", "masked"]
        for mode in forward_mode:
            # [(bs, num_colors)] * num_hidden_objects
            dist = dists[modes.index(mode)]
            result_dists.append(dist)
        if len(forward_mode) == 1:
            return result_dists[0]

        return result_dists

    # def preprocess(self, obs):
    #     """
    #     :param obs: Batch(obs_i_key: (bs, seq_len, obs_i_shape))
    #     :return o: [[(bs, num_colors)] * num_observables] * seq_len
    #             a: (bs, seq_len, num_observables)
    #             ot: [(bs, seq_len, num_colors)] * (num_observables + num_objects)
    #             r: (bs, seq_len, 1)
    #     """
    #     # [(bs, seq_len, num_colors)] * num_observables
    #     o = [F.one_hot(obs[k].squeeze().long(), i).float()
    #          for k, i in zip(self.o_keys, self.o_inner_dim)]
    #     # [[(bs, num_colors)] * seq_len] * num_observables
    #     o = [torch.unbind(o_i, dim=1) for o_i in o]
    #     # [[(bs, num_colors)] * num_observables] * seq_len
    #     o = [list(o_t) for o_t in zip(*o)]
    #
    #     # (bs, seq_len, num_observables)
    #     a = F.one_hot(obs[self.a_key].squeeze().long(), self.a_inner_dim).float()
    #
    #     # [(bs, seq_len, num_colors)] * (num_observables + num_objects)
    #     ot = [F.one_hot(obs[k].squeeze().long(), i).float()
    #           for k, i in zip(self.ot_keys, self.ot_inner_dim)]
    #     # (bs, seq_len, 1)
    #     r = obs[self.r_key]
    #     return o, a, ot, r

    def preprocess(self, obs):
        """
        :param obs: Batch(obs_i_key: (bs, seq_len, obs_i_shape))
        :return o: (bs, seq_len,  num_observables * num_colors)
                a: (bs, seq_len, num_observables)
                ot: (bs, seq_len, (num_observables + num_objects) * num_colors)
                r: (bs, seq_len, 1)
        """
        # [(bs, seq_len, num_colors)] * num_observables
        o = [F.one_hot(obs[k].squeeze().long(), i).float()
             for k, i in zip(self.o_keys, self.o_inner_dim)]
        # (bs, seq_len, num_observables * num_colors)
        o = torch.cat(o, dim=-1)

        # (bs, seq_len, num_observables)
        a = F.one_hot(obs[self.a_key].squeeze().long(), self.a_inner_dim).float()

        # [(bs, seq_len, num_colors)] * (num_observables + num_objects)
        ot = [F.one_hot(obs[k].squeeze().long(), i).float()
              for k, i in zip(self.ot_keys, self.ot_inner_dim)]
        # (bs, seq_len, (num_observables + num_colors) * num_colors)
        ot = torch.cat(ot, dim=-1)

        # (bs, seq_len, 1)
        r = obs[self.r_key]
        return o, a, ot, r

    def postprocess(self, x, z, u):
        """
        :param x: (bs, seq_len, num_observables * num_colors)
        :param z: (bs, seq_len, num_colors)
        :return xz: [[(bs, num_colors)] * num_objects] * seq_len
                u: [(bs, num_observables)] * seq_len
        """
        xz = torch.cat((x, z), dim=-1)
        xz = torch.unbind(xz, dim=1)
        xz = [torch.split(xz_i, self.num_colors, dim=-1) for xz_i in xz]
        u = torch.unbind(u, dim=1)
        return xz, u

    def reparam(self, logits):
        if self.training:
            sample = F.gumbel_softmax(logits, tau=self.gumbel_temp, hard=True)
        else:
            sample = F.one_hot(torch.argmax(logits, dim=-1), logits.size(-1)).float()
        return sample

    def forward(self, obs, forward_with_feature=None):
        """
        if use_all_past:
            History encoder
        else:
            Full DVAE encoder: with/without markovian past (unshared forward MLP),
            fixed future step (last step output of shared RNN) or
            all future (each step output of common RNN)
        :param obs: Batch(obs_i_key: (bs, seq_len, obs_i_shape))
        :return z / z_probs: (bs, seq_len, num_hiddens * num_colors)
                x: (bs, seq_len, num_observables * num_colors)
                u: (bs, seq_len, num_observables)
                st: [(bs, seq_len - num_hiddes, num_colors)] * (2 * num_objects)
                r: (bs, seq_len - num_hiddes, 1)
                eps: (bs, seq_len, num_objects * eps_dim)
                eps_xzu_probs: (bs, seq_len, num_objects * eps_dim)
                xuz_llh: (bs * (seq_len - 2), )
        """
        self.update_num += 1
        o, a, ot, r = self.preprocess(obs)
        x = o
        u = torch.zeros_like(a)
        u[:, 1:] = a[:, :-1]

        bs, seq_len = a.shape[:2]
        # t = 1 to (T - num_hiddes)
        z = torch.zeros(bs, seq_len, self.z_dim).to(self.device)
        z_probs = torch.zeros(bs, seq_len, self.z_dim).to(self.device)
        eps = torch.zeros(bs, seq_len, self.num_objects * self.eps_dim).to(self.device)
        eps_xzu_probs = torch.zeros(bs, seq_len, self.num_objects * self.eps_dim).to(self.device)
        # eps_zeros_probs = torch.zeros(bs, seq_len - self.num_hidden_objects + 1,
        #                               self.num_objects * self.eps_dim).to(self.device)

        # x_{t:T}, u_{t:T} -> g_t;
        # z_{t-1}, x_{t-1:t:2}, u_{t-1} -> m_t;
        # m_t, g_t -> n_t; z_t ~ Categorical(n_t)

        # if self.use_all_future or self.use_all_past:
        #     # x_ids = torch.eye(seq_len).unsqueeze(0).repeat(bs, 1, 1).to(self.device)
        #     # (bs, seq_len, 2 * num_observables * num_colors + num_observables + seq_len)
        #     # xxui = torch.cat((x_tm1, x, u, x_ids), -1)
        #     # xxu = torch.cat((x_tm1, x, u), -1)
        #     # xu = torch.cat((x_tm1, u), -1) if self.use_all_future else torch.cat((x, a), -1)
        #     # xu = torch.cat((x[:, :, :2 * self.num_colors], a[:, :, :2]), -1)
        #     xu = torch.cat((x, a), -1)
        #     g_h = []
        #     for j in range(self.num_hidden_objects):
        #         g, _ = self.rnn_g[j](torch.flip(xu, [1]) if self.use_all_future else xu)
        #         g_h.append(torch.flip(g, [1]) if self.use_all_future else g)

        # x_ids = torch.eye(seq_len).unsqueeze(0).repeat(bs, 1, 1).to(self.device)
        # (bs, seq_len, 2 * num_observables * num_colors + num_observables + seq_len)
        # xxui = torch.cat((x_tm1, x, u, x_ids), -1)
        # xxu = torch.cat((x_tm1, x, u), -1)
        # xu = torch.cat((x_tm1, u), -1) if self.use_all_future else torch.cat((x, a), -1)
        # xu = torch.cat((x[:, :, :2 * self.num_colors], a[:, :, :2]), -1)
        xu = torch.cat((x, a), -1)
        if self.use_all_past:
            g_h_f = []
            for j in range(self.num_hidden_objects):
                g, _ = self.rnn_g[j](xu)
                g_h_f.append(g)

        if self.use_all_future:
            g_h_b = []
            for j in range(self.num_hidden_objects):
                g, _ = self.rnn_g[j](torch.flip(xu, [1]))
                g_h_b.append(torch.flip(g, [1]))

        # idx_one_hot = F.one_hot(
        #     torch.arange(0, seq_len - self.num_hidden_objects).repeat(bs, 1)
        # ).float().to(self.device)
        # for i in range(1, 4):
        for i in range(1, seq_len - self.num_hidden_objects + 1):
            if self.num_future > 0:
                # x_ids = F.one_hot(torch.tensor(i), seq_len).unsqueeze(0).unsqueeze(0)
                # # (bs, num_hiddens, seq_len)
                # x_ids = x_ids.repeat(bs, self.num_hidden_objects, 1).to(self.device)

                # (bs, num_hiddens, 2 * num_observables * num_colors + num_observables + seq_len)
                # xxui = torch.cat((x_tm1[:, i: (i + self.num_hidden_objects)],
                #                  x[:, i: (i + self.num_hidden_objects)],
                #                  u[:, i: (i + self.num_hidden_objects)], x_ids), -1)
                # xxu = torch.cat((x[:, (i - 1):i],
                #                  x[:, i:(i + 1)],
                #                  u[:, i:(i + 1)]), -1)
                xxu = torch.cat((x[:, (i - 1)], x[:, i], u[:, i]), -1)
                # x3u2 = torch.cat((x[:, (i - 1):i],
                #                  x[:, i:(i + 1)],
                #                  x[:, (i + 1):(i + 2)],
                #                  u[:, i:(i + 1)],
                #                  u[:, (i + 1):(i + 2)]), -1)
            for j in range(self.num_hidden_objects):
                # if (self.num_past > 0 and i > 1):
                #     zxu = torch.cat((z[:, i - 1], x[:, i - 2], u[:, i - 1]), -1)
                #     m = self.mlp_m[j](zxu)

                # Past, future, and fixed future encoders
                if self.use_all_past and not self.use_all_future:
                    if self.num_future == 0:
                        n = self.cf_n[0][j](g_h_f[j][:, i - 1])
                    else:
                        n = self.cf_n[0][j](
                            torch.cat((g_h_f[j][:, i - 1], self.mlp_m[j](xxu)), -1)
                        )

                if not self.use_all_past and self.use_all_future:
                    n = self.cf_n[0][j](g_h_b[j][:, i - 1])

                if self.use_all_past and self.use_all_future:
                    # n = self.cf_n[0][j](g_h[j][:, i - 1])
                    # n = self.cf_n[0][j](
                    #     torch.cat((g_h_f[j][:, i - 1], g_h_b[j][:, i - 1], idx_one_hot[:, i - 1]), -1)
                    # )
                    n = self.cf_n[0][j](
                        torch.cat((g_h_f[j][:, i - 1], g_h_b[j][:, i - 1]), -1)
                    )

                # elif self.use_all_future:
                #     mg = torch.cat((m, g_h[j][:, i - 1]), -1) if (self.num_past > 0 and i > 1) \
                #         else g_h[j][:, i - 1]
                #     # mg = (m + g_h[j][:, i - 1]) / 2 if (self.num_past > 0 and i > 1) \
                #     #     else g_h[j][:, i - 1]
                #     n = self.cf_n[1][j](mg) if (self.num_past > 0 and i > 1) else self.cf_n[0][j](mg)
                # elif self.num_future > 0:
                #     g, _ = self.rnn_g[j](torch.flip(xxu, [1]))
                #     # g, _ = self.rnn_g[j](torch.flip(x3u2, [1]))
                #     g = torch.flip(g, [1])
                #     mg = torch.cat((m, g[:, 0]), -1) if (self.num_past > 0 and i > 1) else g[:, 0]
                #     # mg = (m + g[:, 0]) / 2 if (self.num_past > 0 and i > 1) else g[:, 0]
                #     n = self.cf_n[1][j](mg) if (self.num_past > 0 and i > 1) else self.cf_n[0][j](mg)
                # else:
                #     raise ValueError("Invalid encoder type")
                z[:, i, j * self.num_colors: (j + 1) * self.num_colors] = self.reparam(n)
                z_probs[:, i, j * self.num_colors: (j + 1) * self.num_colors] = F.softmax(n / 1, dim=-1)

        # # state + target / reward: t=1 to (T - num_hiddes)
        # # (bs, seq_len - num_hiddes, 2 * num_objects * num_colors)
        # st = torch.cat(
        #     (
        #         ot[:, :-self.num_hidden_objects, :(self.hidden_objects_ind[0] * self.num_colors)],
        #         z[:, 1:((-self.num_hidden_objects + 1) if self.num_hidden_objects > 1 else None)],
        #         ot[:, :-self.num_hidden_objects, (self.hidden_objects_ind[0] * self.num_colors):]
        #     ),
        #     dim=-1
        # )
        #
        # # [(bs, seq_len - num_hiddes, num_colors)] * (2 * num_objects)
        # st = torch.split(st, self.num_colors, dim=-1)
        # r = r[:, :-self.num_hidden_objects]

        # # For history-based encoder
        # # state + target / reward: t=2 to (T - num_hiddes)
        # # (bs, seq_len - num_hiddes - 1, 2 * num_objects * num_colors)
        # st = torch.cat(
        #     (
        #         ot[:, 1:-self.num_hidden_objects, :(self.hidden_objects_ind[0] * self.num_colors)],
        #         z[:, 2:((-self.num_hidden_objects + 1) if self.num_hidden_objects > 1 else None)],
        #         ot[:, 1:-self.num_hidden_objects, (self.hidden_objects_ind[0] * self.num_colors):]
        #     ),
        #     dim=-1
        # )
        #
        # # [(bs, seq_len - num_hiddes, num_colors)] * (2 * num_objects)
        # st = torch.split(st, self.num_colors, dim=-1)
        # r = r[:, 1:-self.num_hidden_objects]


        # state + target / reward: t=2 to (T - num_hiddes)
        # (bs, seq_len - num_hiddes - 1, 2 * num_objects * num_colors)
        st = torch.cat(
            (
                ot[:, (-self.num_hidden_objects - 2):-self.num_hidden_objects, :(self.hidden_objects_ind[0] * self.num_colors)],
                z[:, (-self.num_hidden_objects - 1):((-self.num_hidden_objects + 1) if self.num_hidden_objects > 1 else None)],
                ot[:, (-self.num_hidden_objects - 2):-self.num_hidden_objects, (self.hidden_objects_ind[0] * self.num_colors):]
            ),
            dim=-1
        )

        # [(bs, seq_len - num_hiddes - 1, num_colors)] * (2 * num_objects)
        st = torch.split(st, self.num_colors, dim=-1)
        r = r[:, (-self.num_hidden_objects - 2):-self.num_hidden_objects]

        if not self.eps_encoder:
            return z, z_probs, x, u, st, r
        else:
            # z_detach = z.detach()
            z_detach = z
            xzu_tm1 = torch.cat((x_tm1, z_detach, u), -1)
            xz = torch.cat((x[:, :-1, :self.hidden_objects_ind[0] * self.num_colors],
                            z_detach[:, 1:], x[:, :-1, self.hidden_objects_ind[0] * self.num_colors:]), -1)
            for i in range(1, seq_len - 1):
                for j in range(self.num_objects):
                    xzuxz = torch.cat((xzu_tm1[:, i: (i + 1)],
                                       xz[:, i: (i + 1), j * self.num_colors: (j + 1) * self.num_colors]), -1)
                    g, _ = self.rnn_e[j](torch.flip(xzuxz, [1]))
                    g = torch.flip(g, [1])
                    n = self.mlp_e[j](g[:, 0])
                    # t=1 to T-2
                    eps[:, i, j * self.eps_dim: (j + 1) * self.eps_dim] = self.reparam(n)
                    # eps[:, i, j * self.eps_dim: (j + 1) * self.eps_dim] = n

                    xzu = torch.cat((xzu_tm1[:, i: (i + 1)],
                                     torch.zeros_like(xz[:, i: (i + 1),
                                                      j * self.num_colors: (j + 1) * self.num_colors])), -1)
                    g, _ = self.rnn_e[j](torch.flip(xzu, [1]))
                    g = torch.flip(g, [1])
                    n = self.mlp_e[j](g[:, 0])
                    eps_xzu_probs[:, i, j * self.eps_dim: (j + 1) * self.eps_dim] = F.softmax(n / 1, dim=-1)

                    # all_zeros = torch.zeros_like(xzu)
                    # g, _ = self.rnn_e[j](torch.flip(all_zeros, [1]))
                    # g = torch.flip(g, [1])
                    # n = self.mlp_e[j](g[:, 0])
                    # eps_zeros_probs[:, i, j * self.eps_dim: (j + 1) * self.eps_dim] = F.softmax(n / 1, dim=-1)

            # Batch-wise update the counting-based probability of each tuple (obs, action, hidden)
            # and compute the likelihood of each tuple in the batch
            if self.count_transitions:
                # self.xuz_counts = Counter()
                # self.xuz_probs = {}
                oa_tuple = obs_batch2tuple(obs, self.params)
                h_tuple = hidden_batch2tuple(z)
                oah_batch = [oa + (h,) for oa, h in zip(oa_tuple, h_tuple)]
                mask, oah_batch_uni = rm_dup(oah_batch)

                # [(bs, seq_len - 2, eps_dim)] * num_objects
                eps_xzu_probs = torch.split(eps_xzu_probs[:, 1:-1], self.eps_dim, dim=-1)
                # (bs * (seq_len - 2), num_objects, eps_dim)
                eps_xzu_probs = torch.stack(eps_xzu_probs, dim=-2).view(-1, self.num_objects, self.eps_dim)
                # (bs_uni, num_objects, eps_dim)
                eps_xzu_probs_uni = eps_xzu_probs[mask]

                self.xuz_counts.update(oah_batch)
                self.xuz_probs = count2prob(self.xuz_counts)
                xuz_llh = prob2llh(self.xuz_probs, oah_batch_uni).to(self.device)
                # (bs_uni, 1, 1)
                xuz_probs = xuz_llh.unsqueeze(-1).unsqueeze(-1)
                # (num_objects, eps_dim)
                eps_probs = (eps_xzu_probs_uni * xuz_probs).sum(0)
                # (bs * (seq_len - 2), num_objects, eps_dim)
                eps_probs = eps_probs.unsqueeze(0).expand(len(oah_batch), -1, -1).detach()
            else:
                eps_probs = None

            if not self.update_num % 1000:
                print("xuz_llh", xuz_llh, len(xuz_llh))

            return z, z_probs, x, u, st, r, eps, eps_xzu_probs, eps_probs

    # def forward(self, obs, forward_with_feature):
    #     """
    #     full DVAE encoder with shared markovian forward info; backward RNN at each step
    #     and multi-step prediction
    #     :param obs: Batch(obs_i_key: (bs, seq_len, obs_i_shape))
    #     :return z / z_probs: (bs, seq_len, num_hiddens * num_colors)
    #             x: (bs, seq_len, num_observables * num_colors)
    #             u: (bs, seq_len, num_observables)
    #             st: [(bs, seq_len - 2, num_colors)] * (2 * num_objects)
    #             r: (bs, seq_len - 2, 1)
    #     """
    #     o, a, ot, r = self.preprocess(obs)
    #     x = o
    #     u = torch.zeros_like(a)
    #     u[:, 1:] = a[:, :-1]
    #
    #     bs, seq_len = a.shape[:2]
    #     z_probs = torch.zeros((bs, seq_len, self.z_dim)).to(self.device)
    #     z = torch.zeros((bs, seq_len, self.z_dim)).to(self.device)
    #
    #     # x_{t:T}, u_{t:T} -> g_t;
    #     # z_{t-1}, x_{t-1:t:2}, u_{t-1} -> m_t;
    #     # m_t, g_t -> n_t; z_t ~ Categorical(n_t)
    #     x_tm1 = torch.zeros_like(x)
    #     x_tm1[:, 1:] = x[:, :-1]
    #     # (bs, seq_len, 2 * num_observables * num_colors + num_observables)
    #     xxu = torch.cat((x_tm1, x, u), -1)
    #     g_h = []
    #     for j in range(self.num_hidden_objects):
    #         g, _ = self.rnn_g[j](torch.flip(xxu, [1]))
    #         g = torch.flip(g, [1])
    #         g_h.append(g)
    #     for i in range(1, seq_len):
    #         if i > 1:
    #             # [(bs, 1, num_colors)] * num_hiddens
    #             m = forward_with_feature(x[:, (i - 2):(i - 1)], z[:, (i - 1):i], u[:, (i - 1):i],
    #                                      forward_mode=("full",))[1]
    #         for j in range(self.num_hidden_objects):
    #             n = self.cf_n[0][j](g_h[j][:, i])
    #             if i > 1:
    #                 # (bs, num_colors)
    #                 n = (m[j].squeeze(1) + n) / 2
    #             z[:, i, j * self.num_colors: (j + 1) * self.num_colors] = self.reparam(n)
    #             z_probs[:, i, j * self.num_colors: (j + 1) * self.num_colors] = F.softmax(n / 1, dim=-1)
    #
    #     # state + target / reward: t=1 to T-1
    #     # (bs, seq_len - 1, 2 * num_objects * num_colors)
    #     st = torch.cat((ot[:, :-1, :self.hidden_objects_ind[0] * self.num_colors],
    #                     z[:, 1:], ot[:, :-1, self.hidden_objects_ind[0] * self.num_colors:]), dim=-1)
    #     # [(bs, seq_len - 1, num_colors)] * (2 * num_objects)
    #     st = torch.split(st, self.num_colors, dim=-1)
    #     r = r[:, :-1]
    #     return z, z_probs, x, u, st, r

    # def forward(self, obs):
    #     """
    #     hindsight-based masked MLP encoder
    #     :param obs: Batch(obs_i_key: (bs, seq_len, obs_i_shape))
    #     :return z / z_probs: (bs, seq_len, num_colors)
    #             x: [[(bs, num_colors)] * num_observables] * seq_len
    #             u: [(bs, num_observables)] * seq_len
    #             st: [(bs, seq_len - 2, num_colors)] * (2 * num_objects)
    #             r: (bs, seq_len - 2, 1)
    #     """
    #     o, a, ot, r = self.preprocess(obs)
    #     x = o
    #     u = torch.zeros_like(a)
    #     u[:, 1:] = a[:, :-1]
    #     # [(bs, num_observables)] * seq_len
    #     u = torch.unbind(u, dim=1)
    #
    #     # assume 1 hidden object
    #     bs, seq_len = a.shape[:2]
    #     z_probs = torch.zeros((bs, seq_len, self.num_colors)).to(self.device)
    #     z = torch.zeros((bs, seq_len, self.num_colors)).to(self.device)
    #
    #     # mask = torch.tensor([[0, 1, 0, 1, 1]], dtype=torch.bool).to(self.device)
    #     mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool).to(self.device)
    #     mask = mask.repeat(bs, 1, 1).to(self.device)
    #
    #     for t in range(1, seq_len):
    #         # (bs, num_colors)
    #         z_logit = self.forward_with_feature(x[t - 1] + x[t], u[t], mask=mask, forward_mode=("masked",))[0]
    #         z_probs[:, t] = F.softmax(z_logit / 1, dim=-1)
    #         z[:, t] = self.reparam(z_logit)  # z_t
    #
    #     # state + target / reward: t=1 to T-1
    #     # [(bs, seq_len - 1, num_colors)] * (2 * num_objects)
    #     st = ([ot_i[:, :-1] for ot_i in ot[:self.hidden_objects_ind[0]]] + [z[:, 1:]]
    #           + [ot_i[:, :-1] for ot_i in ot[self.hidden_objects_ind[0]:]])
    #     r = r[:, :-1]
    #     return z, z_probs, x, u, st, r

    # def forward(self, obs):
    #     """
    #     hindsight-based MLP encoder
    #     :param obs: Batch(obs_i_key: (bs, seq_len, obs_i_shape))
    #     :return z / z_probs: (bs, seq_len, num_colors)
    #             x: (bs, seq_len, num_observables * num_colors)
    #             u: (bs, seq_len, num_observables)
    #             st: [(bs, seq_len - 2, num_colors)] * (2 * num_objects)
    #             r: (bs, seq_len - 2, 1)
    #     """
    #     o, a, ot, r = self.preprocess(obs)
    #     x = o
    #     u = torch.zeros_like(a)
    #     u[:, 1:] = a[:, :-1]
    #
    #     # assume 1 hidden object
    #     bs, seq_len = a.shape[:2]
    #     z_probs = torch.zeros((bs, seq_len, self.num_colors)).to(self.device)
    #     z = torch.zeros((bs, seq_len, self.num_colors)).to(self.device)
    #
    #     # x_{t:T}, u_{t:T} -> g_t;
    #     # z_{t-1}, x_{t-1:t:2}, u_{t-1} -> m_t;
    #     # m_t, g_t -> n_t; z_t ~ Categorical(n_t)
    #
    #     # x_tm1 = torch.zeros_like(x)
    #     # x_tm1[:, 1:] = x[:, :-1]
    #     # u_tm1 = torch.zeros_like(u)
    #     # u_tm1[:, :-1] = u[:, 1:]
    #     # xu = torch.cat((x, u_tm1), -1)
    #     # xu = torch.cat((x[:, :, 2 * self.num_colors: 3 * self.num_colors],
    #     #                 x_tm1[:, :, 2 * self.num_colors: 3 * self.num_colors], u), -1)
    #     # g, _ = self.rnn_g(torch.flip(xu, [1]))
    #     # g = torch.flip(g, [1])
    #     for t in range(1, seq_len):
    #         # if t == 1:
    #         #     # m = torch.zeros((bs, self.dim_rnn_g)).to(self.device)
    #         #     # zxu = torch.cat((z[:, t - 1], 0 * x[:, t - 1], u[:, t - 1]), -1)
    #         #     # z_probs[:, t] = F.one_hot(torch.full((bs,), 1), self.num_colors).float()
    #         #     # z[:, t] = F.one_hot(torch.full((bs,), 1), self.num_colors).float()
    #         # else:
    #         #     # zxu = torch.cat((z[:, t - 1], x[:, t - 2], u[:, t - 1]), -1)
    #         #     zxu = torch.cat((z[:, t - 1], x[:, t - 2], u[:, t - 1]), -1)
    #
    #         # (bs, 2 * num_observables * num_colors + num_observables)
    #         xxu = torch.cat((x[:, t - 1], x[:, t], u[:, t]), -1)
    #
    #         n = self.mlp_m(xxu)
    #         # mg = (m + g[:, t]) / 2
    #         # mg = g[:, t-1]
    #         # mg = torch.cat((m, g[:, t]), -1)
    #         # (bs, num_colors)
    #         # n = self.cf_n(m)
    #         z_probs[:, t] = F.softmax(n / 1, dim=-1)
    #         z[:, t] = self.reparam(n)  # z_t
    #
    #     # state + target / reward: t=1 to T-1
    #     # [(bs, seq_len - 1, num_colors)] * (2 * num_objects)
    #     st = ([ot_i[:, :-1] for ot_i in ot[:self.hidden_objects_ind[0]]] + [z[:, 1:]]
    #           + [ot_i[:, :-1] for ot_i in ot[self.hidden_objects_ind[0]:]])
    #     r = r[:, :-1]
    #     return z, z_probs, x, u, st, r

    # def forward(self, obs):
    #     """
    #     hindsight-based MLP encoder for connected hiddens
    #     :param obs: Batch(obs_i_key: (bs, seq_len, obs_i_shape))
    #     :return z / z_probs: (bs, seq_len, num_hiddens * num_colors)
    #             x: (bs, seq_len, num_observables * num_colors)
    #             u: (bs, seq_len, num_observables)
    #             st: [(bs, seq_len - 2, num_colors)] * (2 * num_objects)
    #             r: (bs, seq_len - 2, 1)
    #     """
    #     o, a, ot, r = self.preprocess(obs)
    #     x = o
    #     u = torch.zeros_like(a)
    #     u[:, 1:] = a[:, :-1]
    #
    #     # assume 1 hidden object
    #     bs, seq_len = a.shape[:2]
    #     z_probs = torch.zeros((bs, seq_len, self.z_dim)).to(self.device)
    #     z = torch.zeros((bs, seq_len, self.z_dim)).to(self.device)
    #
    #     for i in range(1, seq_len):
    #         # (bs, 2 * num_observables * num_colors + num_observables)
    #         xxu = torch.cat((x[:, i - 1], x[:, i], u[:, i]), -1)
    #         n = self.mlp_m(xxu)
    #         z_probs[:, i, self.num_colors:] = F.softmax(n / 1, dim=-1)
    #         z[:, i, self.num_colors:] = self.reparam(n)  # h^1_i
    #         if i >= 2:
    #             xz = torch.cat((x[:, i - 2], x[:, i - 1],
    #                             z[:, i - 1, self.num_colors:], z[:, i, self.num_colors:]), -1)
    #             n = self.mlp_m1(xz)
    #             z_probs[:, i - 1, :self.num_colors] = F.softmax(n / 1, dim=-1)
    #             z[:, i - 1, :self.num_colors] = self.reparam(n)  # h^0_{i-1}
    #
    #     # state + target / reward: t=1 to T-2
    #     # (bs, seq_len - 2, 2 * num_objects * num_colors)
    #     st = torch.cat((ot[:, :-2, :self.hidden_objects_ind[0] * self.num_colors],
    #                     z[:, 1:-1], ot[:, :-2, self.hidden_objects_ind[0] * self.num_colors:]), dim=-1)
    #     # [(bs, seq_len - 2, num_colors)] * (2 * num_objects)
    #     st = torch.split(st, self.num_colors, dim=-1)
    #     r = r[:, :-2]
    #     return z, z_probs, x, u, st, r

    # def forward(self, obs):
    #     """
    #     DVAE encoder using RNN for current and future info
    #     :param obs: Batch(obs_i_key: (bs, seq_len, obs_i_shape))
    #     :return z / z_probs: (bs, seq_len, num_colors)
    #             x: (bs, seq_len, num_observables * num_colors)
    #             u: (bs, seq_len, num_observables)
    #             st: [(bs, seq_len - 1, num_colors)] * (2 * num_objects)
    #             r: (bs, seq_len - 1, 1)
    #     """
    #     o, a, ot, r = self.preprocess(obs)
    #     x = o
    #     u = torch.zeros_like(a)
    #     u[:, 1:] = a[:, :-1]
    #
    #     # assume 1 hidden object
    #     bs, seq_len = a.shape[:2]
    #     z_probs = torch.zeros((bs, seq_len, self.z_dim)).to(self.device)
    #     z = torch.zeros((bs, seq_len, self.z_dim)).to(self.device)
    #
    #     # x_{t:T}, u_{t:T} -> g_t;
    #     # z_{t-1}, x_{t-1:t:2}, u_{t-1} -> m_t;
    #     # m_t, g_t -> n_t; z_t ~ Categorical(n_t)
    #
    #     x_tm1 = torch.zeros_like(x)
    #     x_tm1[:, 1:] = x[:, :-1]
    #     # (bs, seq_len, 2 * num_observables * num_colors + num_observables)
    #     xxu = torch.cat((x_tm1, x, u), -1)
    #     g, _ = self.rnn_g(torch.flip(xxu, [1]))
    #
    #     # xu = torch.cat((x, u), -1)
    #     # g, _ = self.rnn_g(torch.flip(xu, [1]))
    #
    #     g = torch.flip(g, [1])
    #     for i in range(1, seq_len):
    #         # if i == 1:
    #         #     m = torch.zeros((bs, self.dim_rnn_g)).to(self.device)
    #         # else:
    #         #     zxu = torch.cat((z[:, i - 1], x[:, i - 2], u[:, i - 1]), -1)
    #         #     m = self.mlp_m(zxu)
    #         # mg = (m + g[:, i]) / 2
    #         mg = g[:, i]
    #         # mg = torch.cat((m, g[:, i]), -1)
    #         # (bs, z_dim)
    #         n = self.cf_n(mg)
    #         # [(bs, num_colors)] * num_hiddens
    #         n = torch.split(n, self.num_colors, dim=-1)
    #
    #         z_i = [self.reparam(n_i) for n_i in n]
    #         z[:, i] = torch.cat(z_i, dim=-1)
    #         z_probs_i = [F.softmax(n_i / 1, dim=-1) for n_i in n]
    #         z_probs[:, i] = torch.cat(z_probs_i, dim=-1)
    #
    #     # state + target / reward: t=1 to T-1
    #     # (bs, seq_len - 1, 2 * num_objects * num_colors)
    #     st = torch.cat((ot[:, :-1, :self.hidden_objects_ind[0] * self.num_colors],
    #                     z[:, 1:], ot[:, :-1, self.hidden_objects_ind[0] * self.num_colors:]), dim=-1)
    #     # [(bs, seq_len - 1, num_colors)] * (2 * num_objects)
    #     st = torch.split(st, self.num_colors, dim=-1)
    #     r = r[:, :-1]
    #     return z, z_probs, x, u, st, r

    # def forward(self, obs):
    #     """
    #     DVAE encoder using RNN for future info and MLP for current info
    #     :param obs: Batch(obs_i_key: (bs, seq_len, obs_i_shape))
    #     :return z / z_probs: (bs, seq_len, z_dim)
    #             x: (bs, seq_len, num_observables * num_colors)
    #             u: (bs, seq_len, num_observables)
    #             st: [(bs, seq_len - 1, num_colors)] * (2 * num_objects)
    #             r: (bs, seq_len - 1, 1)
    #     """
    #     o, a, ot, r = self.preprocess(obs)
    #     x = o
    #     u = torch.zeros_like(a)
    #     u[:, 1:] = a[:, :-1]
    #
    #     # assume 1 hidden object
    #     bs, seq_len = a.shape[:2]
    #     z_probs = torch.zeros((bs, seq_len, self.z_dim)).to(self.device)
    #     z = torch.zeros((bs, seq_len, self.z_dim)).to(self.device)
    #
    #     # x_{t:T}, u_{t:T} -> g_t;
    #     # z_{t-1}, x_{t-1:t:2}, u_{t-1} -> m_t;
    #     # m_t, g_t -> n_t; z_t ~ Categorical(n_t)
    #
    #     # x_tm1 = torch.zeros_like(x)
    #     # x_tm1[:, 1:] = x[:, :-1]
    #     # # (bs, seq_len, 2 * num_observables * num_colors + num_observables)
    #     # xxu = torch.cat((x_tm1, x, u), -1)
    #     # g, _ = self.rnn_g(torch.flip(xxu, [1]))
    #
    #     xu = torch.cat((x, a), -1)
    #     g, _ = self.rnn_g(torch.flip(xu, [1]))
    #
    #     g = torch.flip(g, [1])
    #     for i in range(1, seq_len):
    #         # if i == 1:
    #         #     m = torch.zeros((bs, self.dim_rnn_g)).to(self.device)
    #         # else:
    #         #     zxu = torch.cat((z[:, i - 1], x[:, i - 2], u[:, i - 1]), -1)
    #         #     m = self.mlp_m(zxu)
    #         # mg = (m + g[:, i]) / 2
    #         # mg = g[:, i]
    #         xu_i = torch.cat((x[:, i-1], u[:, i]), -1)
    #         m = self.mlp_m(xu_i)
    #         mg = torch.cat((m, g[:, i]), -1)
    #         # (bs, z_dim)
    #         n = self.cf_n(mg)
    #         # [(bs, num_colors)] * num_hiddens
    #         n = torch.split(n, self.num_colors, dim=-1)
    #
    #         z_i = [self.reparam(n_i) for n_i in n]
    #         z[:, i] = torch.cat(z_i, dim=-1)
    #         z_probs_i = [F.softmax(n_i / 1, dim=-1) for n_i in n]
    #         z_probs[:, i] = torch.cat(z_probs_i, dim=-1)
    #
    #     # state + target / reward: t=1 to T-1
    #     # (bs, seq_len - 1, 2 * num_objects * num_colors)
    #     st = torch.cat((ot[:, :-1, :self.hidden_objects_ind[0] * self.num_colors],
    #                     z[:, 1:], ot[:, :-1, self.hidden_objects_ind[0] * self.num_colors:]), dim=-1)
    #     # [(bs, seq_len - 1, num_colors)] * (2 * num_objects)
    #     st = torch.split(st, self.num_colors, dim=-1)
    #     r = r[:, :-1]
    #     return z, z_probs, x, u, st, r