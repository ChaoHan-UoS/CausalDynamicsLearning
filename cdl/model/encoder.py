import sys

from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical

from model.inference_utils import reset_layer, forward_network, forward_network_batch, \
    obs_batch_dict2tuple, count2prob, count2prob_llh


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

            # second lstm layer
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
            obs_act, obs_rew = obs[-2].clone(), obs[-1].clone()  # (bs, stack_num, (n_pred_step), action_dim / reward_dim)
            # mask out the last action / reward in the stacked obs
            # obs_act[:, -1], obs_rew[:, -1] = obs_act[:, -1] * 0, obs_rew[:, -1] * 0
            obs_act[:, -1], obs_rew = obs_act[:, -1] * 0, obs_rew * 0
            # obs_act, obs_rew = obs_act * 0, obs_rew * 0

            obs_obs_dims = len(obs_obs.shape)
            if obs_obs_dims == 5:
                obs_obs = obs_obs.permute(1, 2, 0, 3, 4)  # (bs, stack_num, num_dset_objects, n_pred_step, num_colors)
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
                obs_pred_step = [torch.cat((obses_i, acts_i, rews_i), dim=-1)
                                 for obses_i, acts_i, rews_i in
                                 zip(obs_obs_pred_step, obs_act_pred_step, obs_rew_pred_step)]

                obs_enc = []
                for obs_i in obs_pred_step:
                    obs_enc_i, s_n = super().forward(obs_i, s_0)  # obs_enc with shape (bs, logit_shape)
                    # s_0 = s_n

                    obs_enc_i = obs_enc_i.reshape(-1, len(self.hidden_ind), self.num_colors)  # (bs, num_hidden_states, num_colors)
                    obs_enc_i = F.gumbel_softmax(obs_enc_i, hard=True) if self.training \
                        else F.one_hot(torch.argmax(obs_enc_i, dim=-1), obs_enc_i.size(-1)).float()
                    obs_enc.append(obs_enc_i)  # [(bs, num_hidden_states, num_colors)] * n_pred_step
                obs_enc = torch.stack(obs_enc, dim=-2)  # (bs, num_hidden_states, n_pred_step, num_colors)
            else:
                obs_obs = obs_obs.permute(1, 2, 0, 3)  # (bs, stack_num, num_dset_observables, num_colors)
                obs_obs = obs_obs.reshape(obs_obs.shape[:2] + (-1,))  # (bs, stack_num, num_dset_observables * num_colors)

                # concatenate one-hot observables, action and scalar reward along the last dim
                # (bs, stack_num, num_dset_observables * num_colors + action_dim + reward_dim)
                obs = torch.cat((obs_obs, obs_act, obs_rew), dim=-1)

                obs_enc, s_n = super().forward(obs, s_0)  # obs_enc with shape (bs, logit_shape)
                obs_enc = obs_enc.reshape(-1, len(self.hidden_ind), self.num_colors)  # (bs, num_hidden_states, num_colors)
                obs_enc = F.gumbel_softmax(obs_enc, hard=True) if self.training \
                    else F.one_hot(torch.argmax(obs_enc, dim=-1), obs_enc.size(-1)).float()
            obs_enc = torch.unbind(obs_enc, dim=1)  # [(bs, (n_pred_step), num_colors)] * num_hidden_states

            """concatenate outputs of FNN and RNN"""
            num_hidden_objects = len(self.hidden_objects_ind)
            num_obs_objects = self.num_objects - len(self.hidden_objects_ind)
            if len(self.hidden_targets_ind) > 0:
                obs_enc = obs_obs_forward[:self.hidden_objects_ind[0]] + obs_enc[:num_hidden_objects] \
                          + obs_obs_forward[self.hidden_objects_ind[0]:(num_obs_objects + self.hidden_targets_ind[0])] \
                          + obs_enc[num_hidden_objects:] + obs_obs_forward[(num_obs_objects + self.hidden_targets_ind[0]):]
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
                self.state_feature_1st_layer_weights.append(nn.Parameter(torch.zeros(num_hidden_objects, in_dim, out_dim)))
                self.state_feature_1st_layer_biases.append(nn.Parameter(torch.zeros(num_hidden_objects, 1, out_dim)))
            in_dim = out_dim

        for out_dim in fc_dims:
            self.state_feature_weights.append(nn.Parameter(torch.zeros(num_hidden_objects * feature_dim_dset, in_dim, out_dim)))
            self.state_feature_biases.append(nn.Parameter(torch.zeros(num_hidden_objects * feature_dim_dset, 1, out_dim)))
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
            full_sa_feature, full_sa_indices = full_sa_feature.max(dim=1)           # (num_hidden_objects, bs, out_dim)

            # 3. predict the distribution of next time step value
            full_dist = self.predict_from_sa_feature(full_sa_feature, full_feature)  # [(bs, num_colors)] * num_hidden_objects

        if forward_masked:
            # 1. extract features of all state variables
            # (num_hidden_objects, feature_dim_dset, bs, out_dim)
            masked_state_feature = self.extract_state_feature(masked_feature)

            # 2. extract global feature by element-wise max
            # mask out unused features
            # (num_hidden_objects, feature_dim_dset + 1, bs, out_dim)
            masked_sa_feature = torch.cat([masked_state_feature, action_feature], dim=1)
            mask = mask.permute(1, 2, 0)                                            # (num_hidden_objects, feature_dim_dset + 1, bs)
            masked_sa_feature[~mask] = float('-inf')
            masked_sa_feature, masked_sa_indices = masked_sa_feature.max(dim=1)     # (num_hidden_objects, bs, out_dim)

            # 3. predict the distribution of next time step value
            masked_dist = self.predict_from_sa_feature(masked_sa_feature, masked_feature)  # [(bs, num_colors)] * num_hidden_objects

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
            mu = torch.stack([dist.mean for dist in dist_list], dim=-2)         # (bs, n_pred_step, feature_dim)
            std = torch.stack([dist.stddev for dist in dist_list], dim=-2)      # (bs, n_pred_step, feature_dim)
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

    def forward(self, obs, feature, action, mask_dset):
        """
        masked MLP fed by d-set
        :param obs: Batch(obs_i_key: (bs, stack_num, (n_pred_step), obs_i_shape))
        :param feature: [[(bs, num_colors)] * (num_dset_observables * (stack_num - 1))] * n_pred_step or
                         [(bs, num_colors)] * (num_dset_observables * (stack_num - 1))
        :param action: [(bs, action_dim)] * n_pred_step or (bs, action_dim)
        :param mask_dset: (num_hidden_objects, feature_dim_dset + 1)
        the second dim represents binary d-sets at t, t+1 and binary action at t
        :return enc_dist/enc_feature: [(bs, (n_pred_step), num_colors)] * num_hidden_objects
                hoa_llh: (bs, ) likelihoods of (hidden, obs, action)
        """
        bs = len(obs[next(iter(dict(obs)))])
        oa_batch = obs_batch_dict2tuple(obs, self.params)[0]

        forward_mode = ("masked",)
        # (bs, num_hidden_objects, feature_dim_dset + 1)
        mask = torch.ones(bs, self.num_hidden_objects, self.feature_dim_dset + 1, dtype=torch.bool,
                          device=self.device)
        mask[:] = mask_dset.bool()  # broadcast along bs

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


def obs_encoder(params):
    encoder_type = params.encoder_params.encoder_type
    device = params.device
    if encoder_type == "recurrent":
        recurrent_enc_params = params.encoder_params.recurrent_enc_params
        chemical_env_params = params.env_params.chemical_env_params
        # obs_shape = len(params.obs_keys) * chemical_env_params.num_colors + params.action_dim + chemical_env_params.max_steps + 1
        obs_shape = len(chemical_env_params.keys_dset) * chemical_env_params.num_colors + params.action_dim + 1
        hidden_layer_size = recurrent_enc_params.hidden_layer_size
        layer_num = recurrent_enc_params.layer_num
        # logit_shape = chemical_env_params.num_objects * chemical_env_params.num_colors
        logit_shape = len(params.hidden_ind) * chemical_env_params.num_colors  # one-hot colors of hidden objects
        encoder = RecurrentEncoder(params, obs_shape, hidden_layer_size, layer_num, logit_shape).to(device)
    elif encoder_type == "feedforward":
        encoder = ForwardEncoder(params).to(device)
    else:
        raise ValueError("Unknown encoder_type: {}".format(encoder_type))
    return encoder


def act_encoder(actions_batch, env, params):
    """
    :param actions_batch: (bs, n_pred_step, action_dim)
    :return: (bs, n_pred_step, num_objects)
    """
    a_ids = actions_batch.view(-1).to(torch.int64)
    enc_actions_batch = torch.tensor([env.partial_act_dims[idx] for idx in a_ids],
                                     dtype=torch.int64, device=params.device)
    enc_actions_batch = enc_actions_batch.view(actions_batch.shape)
    return enc_actions_batch
