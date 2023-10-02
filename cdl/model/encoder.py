import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.inference_utils import reset_layer


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


class MLP(nn.Module):
    def __init__(self, params, obs_shape, hidden_layer_size, layer_num, logit_shape):
        super().__init__()
        dropout = params.encoder_params.feedforward_enc_params.dropout
        self.device = params.device
        self.obs_shape = obs_shape
        self.hidden_layer_size = hidden_layer_size

        self.feedforward = nn.Sequential(
            nn.Linear(obs_shape, hidden_layer_size),
            nn.LayerNorm(hidden_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LayerNorm(hidden_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_layer_size, logit_shape)
        )

        # apply the same initialization scheme to linear layers in the encoder
        # as those in the transition model
        for layer in self.feedforward:
            if isinstance(layer, nn.Linear):
                reset_layer(layer.weight, layer.bias)

    def forward(self, obs, s_0=None):
        """
        :param obs: (bs, obs_shape)
        :return: logit: (bs, logit_shape)
        """
        logit = self.feedforward(obs)  # (bs, logit_shape)
        return logit, s_0


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


class ForwardEncoder(MLP):
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)

        chemical_env_params = params.env_params.chemical_env_params
        self.hidden_objects_ind = chemical_env_params.hidden_objects_ind
        self.hidden_targets_ind = chemical_env_params.hidden_targets_ind
        self.hidden_ind = params.hidden_ind
        self.num_objects = params.env_params.chemical_env_params.num_objects
        self.num_colors = params.env_params.chemical_env_params.num_colors
        self.gumbel_temp = params.encoder_params.feedforward_enc_params.gumbel_temp_0

        # d-set keys and all observable keys
        self.keys_dset = params.env_params.chemical_env_params.keys_dset
        self.keys_remapped_dset = self.keys_dset + ['act'] + ['rew']
        self.keys = [key for key in params.obs_keys if params.obs_spec[key].ndim == 1]

        # input features of encoder, where d-set features are fed to MLP
        self.feature_dim_p = np.sum([len(params.obs_spec[key]) for key in self.keys])
        self.continuous_state = params.continuous_state
        self.feature_inner_dim_p = None
        if not self.continuous_state:
            self.feature_inner_dim_p_dset = np.concatenate([params.obs_dims[key] for key in self.keys_dset])
            self.feature_inner_dim_p = np.concatenate([params.obs_dims[key] for key in self.keys])
        self.feature_inner_dim_remapped_p_dset = np.append(self.feature_inner_dim_p_dset, [params.action_dim, 1])

        # output features of encoder
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
            obs_obs_forward = torch.unbind(obs_obs_forward[:, :, -2])  # [(bs, (n_pred_step), num_colors)] * num_observables

            # identity encoder in the fully observable
            if len(self.hidden_ind) == 0:
                obs_enc = list(obs_obs_forward)  # [(bs, (n_pred_step), num_colors)] * (2 * num_objects)
                return obs_enc[:3], obs_enc[3:]

            """MLP fed by d-set"""
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
                obs_pred_step = [obs_pred_step_i[:, 1:]
                                 for obs_pred_step_i in obs_pred_step]
                # [(bs, (stack_num - 1) * (num_dset_observables * num_colors + action_dim + reward_dim))] * n_pred_step
                obs_pred_step = [obs_pred_step_i.reshape(obs_pred_step_i.shape[:1] + (-1,))
                                 for obs_pred_step_i in obs_pred_step]

                obs_enc = []
                for obs_i in obs_pred_step:
                    obs_enc_i, s_n = super().forward(obs_i, s_0)  # obs_enc with shape (bs, logit_shape)
                    # s_0 = s_n

                    obs_enc_i = obs_enc_i.reshape(-1, len(self.hidden_ind), self.num_colors)  # (bs, num_hidden_states, num_colors)
                    obs_enc_i = F.gumbel_softmax(obs_enc_i, tau=self.gumbel_temp, hard=True) if self.training \
                        else F.one_hot(torch.argmax(obs_enc_i, dim=-1), obs_enc_i.size(-1)).float()
                    obs_enc.append(obs_enc_i)  # [(bs, num_hidden_states, num_colors)] * n_pred_step
                obs_enc = torch.stack(obs_enc, dim=-2)  # (bs, num_hidden_states, n_pred_step, num_colors)
            else:
                obs_obs = obs_obs.permute(1, 2, 0, 3)  # (bs, stack_num, num_dset_observables, num_colors)
                obs_obs = obs_obs.reshape(obs_obs.shape[:2] + (-1,))  # (bs, stack_num, num_dset_observables * num_colors)

                # concatenate one-hot observables, action and scalar reward along the last dim
                # (bs, stack_num, num_dset_observables * num_colors + action_dim + reward_dim)
                obs = torch.cat((obs_obs, obs_act, obs_rew), dim=-1)
                obs = obs[:, 1:]
                # (bs, (stack_num - 1) * (num_dset_observables * num_colors + action_dim + reward_dim))
                obs = obs.reshape(obs.shape[:1] + (-1,))

                obs_enc, s_n = super().forward(obs, s_0)  # obs_enc with shape (bs, logit_shape)
                obs_enc = obs_enc.reshape(-1, len(self.hidden_ind), self.num_colors)  # (bs, num_hidden_states, num_colors)
                obs_enc = F.gumbel_softmax(obs_enc, tau=self.gumbel_temp, hard=True) if self.training \
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
        replay_buffer_params = params.training_params.replay_buffer_params
        feedforward_enc_params = params.encoder_params.feedforward_enc_params
        chemical_env_params = params.env_params.chemical_env_params
        obs_shape = ((replay_buffer_params.stack_num - 1) *
                     (len(chemical_env_params.keys_dset) * chemical_env_params.num_colors + params.action_dim + 1))
        # obs_shape = len(chemical_env_params.keys_dset) * chemical_env_params.num_colors + params.action_dim + 1
        hidden_layer_size = feedforward_enc_params.hidden_layer_size
        layer_num = feedforward_enc_params.layer_num
        logit_shape = len(params.hidden_ind) * chemical_env_params.num_colors  # one-hot colors of hidden objects
        encoder = ForwardEncoder(params, obs_shape, hidden_layer_size, layer_num, logit_shape).to(device)
    else:
        raise ValueError("Unknown encoder_type: {}".format(encoder_type))
    return encoder


def act_encoder(actions_batch, env, params):
    """
    :param actions_batch: (bs, n_pred_step, action_dim)
    :return: (bs, n_pred_step, num_objects)
    """
    a_ids = actions_batch.view(-1).to(torch.int64)
    actions_batch_enc = torch.tensor([env.partial_act_dims[idx] for idx in a_ids],
                                     dtype=torch.int64, device=params.device)
    actions_batch_enc = actions_batch_enc.view(actions_batch.shape)

    return actions_batch_enc
