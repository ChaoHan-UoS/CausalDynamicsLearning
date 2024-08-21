import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.distribution import Distribution
from torch.distributions.one_hot_categorical import OneHotCategorical

from model.inference import Inference
from model.inference_utils import reset_layer, forward_network, forward_network_batch, get_state_abstraction
from utils.utils import to_numpy, preprocess_obs, postprocess_obs


class InferenceCMI(Inference):
    def __init__(self, encoder, decoder, params):
        self.cmi_params = params.inference_params.cmi_params
        self.init_graph(params, encoder)
        super(InferenceCMI, self).__init__(encoder, decoder, params)
        self.causal_pred_reward_mean = 0
        self.causal_pred_reward_std = 1
        self.pred_diff_reward_std = 1

        self.init_abstraction()
        self.init_cache()
        self.reset_causal_graph_eval()

        self.update_num = 0
        self.update_num_eval = 0
        self.print_eval_freq = 10

    def init_model(self, encoder):
        params = self.params
        cmi_params = self.cmi_params

        chemical_env_params = params.env_params.chemical_env_params
        self.num_colors = chemical_env_params.num_colors
        self.hidden_objects_ind = chemical_env_params.hidden_objects_ind
        self.obs_objects_ind = [i for i in range(chemical_env_params.num_objects) if i not in self.hidden_objects_ind]

        # model params
        continuous_state = self.continuous_state

        self.action_dim = action_dim = params.action_dim
        self.feature_dim = feature_dim = encoder.feature_dim
        self.feature_inner_dim = encoder.feature_inner_dim

        self.action_feature_weights = nn.ParameterList()  # [(feature_dim * 1, in_dim_i, out_dim_i)] * len(feature_fc_dims)
        self.action_feature_biases = nn.ParameterList()  # [(feature_dim * 1, 1, out_dim_i)] * len(feature_fc_dims)
        self.state_feature_weights = nn.ParameterList()  # (feature_dim * feature_dim, in_dim_i, out_dim_i) * len(feature_fc_dims[1:]) for discrete state space
        self.state_feature_biases = nn.ParameterList()  # (feature_dim * feature_dim, 1, out_dim_i) * len(feature_fc_dims[1:]) for discrete state space
        self.generative_weights = nn.ParameterList()  # [(feature_dim, in_dim_i, out_dim_i)] * len(generative_fc_dims)
        self.generative_biases = nn.ParameterList()  # [(feature_dim, 1, out_dim_i)] * len(generative_fc_dims)

        # only needed for discrete state space
        self.state_feature_1st_layer_weights = nn.ParameterList()  # [(feature_dim, feature_inner_dim_i, out_dim)] * len(feature_inner_dim)
        self.state_feature_1st_layer_biases = nn.ParameterList()  # [(feature_dim, 1, out_dim)] * len(feature_inner_dim)
        self.generative_last_layer_weights = nn.ParameterList()  # [(1, in_dim, feature_inner_dim_i)] * len(feature_inner_dim)
        self.generative_last_layer_biases = nn.ParameterList()  # [(1, 1, feature_inner_dim_i)] * len(feature_inner_dim)

        # Instantiate the parameters of each layer in the model of each variable at next time step to predict
        # action feature extractor
        in_dim = action_dim
        for out_dim in cmi_params.feature_fc_dims:
            self.action_feature_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
            self.action_feature_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        # state feature extractor
        if continuous_state:
            in_dim = 1
            fc_dims = cmi_params.feature_fc_dims
        else:
            out_dim = cmi_params.feature_fc_dims[0]
            fc_dims = cmi_params.feature_fc_dims[1:]
            for feature_i_dim in self.feature_inner_dim:
                in_dim = feature_i_dim
                self.state_feature_1st_layer_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
                self.state_feature_1st_layer_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        for out_dim in fc_dims:
            self.state_feature_weights.append(nn.Parameter(torch.zeros(feature_dim * feature_dim, in_dim, out_dim)))
            self.state_feature_biases.append(nn.Parameter(torch.zeros(feature_dim * feature_dim, 1, out_dim)))
            in_dim = out_dim

        # predictor
        in_dim = cmi_params.feature_fc_dims[-1]
        for out_dim in cmi_params.generative_fc_dims:
            self.generative_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
            self.generative_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        if continuous_state:
            self.generative_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, 2)))
            self.generative_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, 2)))
        else:
            for feature_i_dim in self.feature_inner_dim:
                final_dim = 2 if feature_i_dim == 1 else feature_i_dim
                self.generative_last_layer_weights.append(nn.Parameter(torch.zeros(1, in_dim, final_dim)))
                self.generative_last_layer_biases.append(nn.Parameter(torch.zeros(1, 1, final_dim)))

    def reset_params(self):
        feature_dim = self.feature_dim
        for w, b in zip(self.action_feature_weights, self.action_feature_biases):
            for i in range(feature_dim):
                reset_layer(w[i], b[i])
        for w, b in zip(self.state_feature_1st_layer_weights, self.state_feature_1st_layer_biases):
            for i in range(feature_dim):
                reset_layer(w[i], b[i])
        for w, b in zip(self.state_feature_weights, self.state_feature_biases):
            for i in range(feature_dim * feature_dim):
                reset_layer(w[i], b[i])
        for w, b in zip(self.generative_weights, self.generative_biases):
            for i in range(feature_dim):
                reset_layer(w[i], b[i])
        for w, b in zip(self.generative_last_layer_weights, self.generative_last_layer_biases):
            reset_layer(w, b)

    def init_graph(self, params, encoder):
        feature_dim = encoder.feature_dim
        device = params.device
        self.stack_num = stack_num = params.training_params.replay_buffer_params.stack_num
        eval_batch_size = self.cmi_params.eval_batch_size
        self.CMI_threshold = self.cmi_params.CMI_threshold

        # used for masking diagonal elements
        # self.diag_mask = torch.eye(feature_dim, feature_dim + 1, dtype=torch.bool, device=device)
        # self.mask_CMI_lb = (torch.ones(feature_dim, feature_dim + 1, device=device) * self.CMI_threshold
        #                     - 0.1 / np.sqrt(eval_batch_size * (stack_num - 2)))
        # self.mask_CMI_ub = (torch.ones(feature_dim, feature_dim + 1, device=device) * self.CMI_threshold
        #                     + 0.1 / np.sqrt(eval_batch_size * (stack_num - 2)))
        # self.mask_CMI_lb = torch.ones(feature_dim, feature_dim + 1, device=device) * self.CMI_threshold
        # self.mask_CMI_ub = torch.ones(feature_dim, feature_dim + 1, device=device) * self.CMI_threshold
        self.mask_CMI = torch.ones(feature_dim, feature_dim + 1, device=device) * self.CMI_threshold
        self.mask = torch.ones(feature_dim, feature_dim + 1, dtype=torch.bool, device=device)
        self.CMI_history = []

    def init_abstraction(self):
        self.abstraction_quested = False
        self.abstraction_graph = None
        self.action_children_idxes = None

    def init_cache(self):
        # cache for faster mask updates
        self.use_cache = False
        self.sa_feature_cache = None
        self.action_feature = None
        self.full_state_feature = None
        self.causal_state_feature = None

        feature_dim = self.feature_dim
        self.feature_diag_mask = torch.eye(feature_dim, dtype=torch.float32, device=self.device)
        self.feature_diag_mask = self.feature_diag_mask.view(feature_dim, feature_dim, 1, 1)

    def reset_causal_graph_eval(self):
        self.mask_update_idx = 0
        # self.eval_step_CMI_lb = torch.zeros(self.feature_dim, self.feature_dim + 1, device=self.device)
        # self.eval_step_CMI_ub = torch.zeros(self.feature_dim, self.feature_dim + 1, device=self.device)
        self.eval_step_CMI = torch.zeros(self.feature_dim, self.feature_dim + 1, device=self.device)

    def extract_action_feature(self, action):
        """
        :param action: (bs, action_dim). notice that bs must be 1D
        :return: (feature_dim, 1, bs, out_dim)
        """
        action = action.unsqueeze(dim=0)  # (1, bs, action_dim)
        action = action.expand(self.feature_dim, -1, -1)  # (feature_dim, bs, action_dim)
        action_feature = forward_network(action,
                                         self.action_feature_weights,
                                         self.action_feature_biases,
                                         dropout=self.dropout)  # (feature_dim, bs, feature_fc_dims[0])
        return action_feature.unsqueeze(dim=1)  # (feature_dim, 1, bs, out_dim)

    def extract_state_feature(self, feature):
        """
        :param feature:
            if state space is continuous: (bs, feature_dim).
            else: [(bs, feature_i_dim)] * feature_dim
            notice that bs must be 1D
        :return: (feature_dim, feature_dim, bs, out_dim),
            the first feature_dim is each state variable at next time step to predict, the second feature_dim are
            inputs (all current state variables) for the prediction
        """
        feature_dim = self.feature_dim
        if self.continuous_state:
            bs = feature.shape[0]
            x = feature.transpose(0, 1)  # (feature_dim, bs)
            x = x.repeat(feature_dim, 1, 1)  # (feature_dim, feature_dim, bs)
            x = x.view(feature_dim * feature_dim, bs, 1)  # (feature_dim * feature_dim, bs, 1)
        else:
            bs = feature[0].shape[0]
            # [(bs, feature_i_dim)] * feature_dim
            reshaped_feature = []
            for f_i in feature:
                f_i = f_i.repeat(feature_dim, 1, 1)  # (feature_dim, bs, feature_i_dim)
                reshaped_feature.append(f_i)
            x = forward_network_batch(reshaped_feature,
                                      self.state_feature_1st_layer_weights,
                                      self.state_feature_1st_layer_biases,
                                      dropout=self.dropout)
            x = torch.stack(x, dim=1)  # (feature_dim, feature_dim, bs, out_dim)
            x = x.view(feature_dim * feature_dim, *x.shape[2:])  # (feature_dim * feature_dim, bs, out_dim)

        state_feature = forward_network(x, self.state_feature_weights, self.state_feature_biases, dropout=self.dropout)
        state_feature = state_feature.view(feature_dim, feature_dim, bs, -1)
        return state_feature  # (feature_dim, feature_dim, bs, out_dim)

    def extract_masked_state_feature(self, masked_feature, full_state_feature):
        """
        :param masked_feature:
            if state space is continuous: (bs, feature_dim).
            else: [(bs, feature_i_dim)] * feature_dim
            notice that bs can be a multi-dimensional batch size
        :param full_state_feature: (feature_dim, feature_dim, bs, out_dim), calculated by self.extract_state_feature()
        :return: (feature_dim, feature_dim, bs, out_dim),
            the first feature_dim is each state variable at next time step to predict, the second feature_dim are
            inputs (all current state variables) for the prediction
        """
        feature_dim = self.feature_dim
        if self.continuous_state:
            x = masked_feature.transpose(0, 1)  # (feature_dim, bs)
            x = x.unsqueeze(dim=-1)  # (feature_dim, bs, 1)
        else:
            # [(1, bs, feature_i_dim)] * feature_dim
            masked_feature = [f_i.unsqueeze(dim=0) for f_i in masked_feature]
            x = forward_network_batch(masked_feature,
                                      [w[i:i + 1] for i, w in enumerate(self.state_feature_1st_layer_weights)],
                                      [b[i:i + 1] for i, b in enumerate(self.state_feature_1st_layer_biases)],
                                      dropout=self.dropout)
            x = torch.cat(x, dim=0)  # (feature_dim, bs, out_dim)

        idxes = [i * (feature_dim + 1) for i in range(feature_dim)]
        x = forward_network(x,
                            [w[idxes] for w in self.state_feature_weights],
                            [b[idxes] for b in self.state_feature_biases],
                            dropout=self.dropout)  # (feature_dim, bs, out_dim)

        feature_diag_mask = self.feature_diag_mask  # (feature_dim, feature_dim, 1, 1)
        masked_state_feature = x.unsqueeze(dim=0)  # (1, feature_dim, bs, out_dim)
        masked_state_feature = full_state_feature * (1 - feature_diag_mask) + masked_state_feature * feature_diag_mask
        # masked_state_feature = masked_state_feature.repeat(feature_dim, 1, 1, 1)
        return masked_state_feature  # (feature_dim, feature_dim, bs, out_dim)

    def predict_from_sa_feature(self, sa_feature, residual_base=None, abstraction_mode=False):
        """
        predict the distribution and sample for the next step value of all state variables
        :param sa_feature: (feature_dim, bs, sa_feature_dim), global feature used for prediction,
            notice that bs can be a multi-dimensional batch size
        :param residual_base: (bs, feature_dim), residual used for continuous state variable prediction
        :param abstraction_mode: if the prediction is computed for state variables in the abstraction only.
            If True, all feature_dim in this function should be replaced by abstraction_feature_dim when indicating
            shapes of tensors.
        :return: next step value for all state variables in the format of distribution,
            if state space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * feature_dim, each of shape (bs, feature_i_dim)
        """
        if abstraction_mode:
            generative_weights = self.abstraction_generative_weights
            generative_biases = self.abstraction_generative_biases
            generative_last_layer_weights = self.abstraction_generative_last_layer_weights
            generative_last_layer_biases = self.abstraction_generative_last_layer_biases
        else:
            generative_weights, generative_biases = self.generative_weights, self.generative_biases
            generative_last_layer_weights = self.generative_last_layer_weights
            generative_last_layer_biases = self.generative_last_layer_biases

        x = forward_network(sa_feature, generative_weights,
                            generative_biases, dropout=self.dropout)  # (feature_dim, bs, out_dim)

        if self.continuous_state:
            x = x.permute(1, 0, 2)  # (bs, feature_dim, 2)
            mu, log_std = x.unbind(dim=-1)  # (bs, feature_dim) * 2
            return self.normal_helper(mu, residual_base, log_std)
        else:
            x = F.relu(x)  # (feature_dim, bs, out_dim)
            x = [x_i.unsqueeze(dim=0) for x_i in torch.unbind(x, dim=0)]  # [(1, bs, out_dim)] * feature_dim
            x = forward_network_batch(x,
                                      generative_last_layer_weights,
                                      generative_last_layer_biases,
                                      activation=None,
                                      dropout=self.dropout)  # [(1, bs, feature_inner_dim_i)] * feature_dim

            feature_inner_dim = self.feature_inner_dim
            if abstraction_mode:
                feature_inner_dim = feature_inner_dim

            dist = []
            for base_i, feature_i_inner_dim, dist_i in zip(residual_base, feature_inner_dim, x):
                dist_i = dist_i.squeeze(dim=0)  # (bs, feature_inner_dim_i)
                if feature_i_inner_dim == 1:
                    mu, log_std = torch.split(dist_i, 1, dim=-1)  # (bs, 1), (bs, 1)
                    dist.append(self.normal_helper(mu, base_i, log_std))
                else:
                    dist.append(OneHotCategorical(logits=dist_i))
                    # dist.append(dist_i)
            return dist

    def forward_step(self, full_feature, masked_feature, causal_feature, action, mask=None,
                     action_feature=None, full_state_feature=None):
        """
        :param full_feature: if state space is continuous: (bs, feature_dim).
            Otherwise: [(bs, feature_i_dim)] * feature_dim
            if it is None, no need to forward it
        :param masked_feature: (bs, feature_dim) or [(bs, feature_i_dim)] * feature_dim
        :param causal_feature: (bs, feature_dim) or [(bs, feature_i_dim)] * feature_dim
        :param action: (bs, action_dim)
        :param mask: (bs, feature_dim, feature_dim + 1)
        :param action_feature: (bs, feature_dim, 1, out_dim), pre-cached value
        :param full_state_feature: (bs, feature_dim, feature_dim, out_dim), pre-cached value
        :param no_causal: not to forward causal_feature, used for training
        :param causal_only: whether to only forward causal_feature, used for curiosity reward and model-based roll-out
        :return: next step value for all state variables in the format of distribution,
            if state space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * feature_dim, each of shape (bs, feature_i_dim)
        """
        forward_full = full_feature is not None
        forward_masked = masked_feature is not None
        forward_causal = causal_feature is not None

        full_dist = masked_dist = causal_dist = None

        if action_feature is None:
            # extract features of the action
            # (bs, action_dim) -> (feature_dim, 1, bs, out_dim)
            self.action_feature = action_feature = self.extract_action_feature(action)

        if forward_full:
            # 1. extract features of all state variables
            if full_state_feature is None:
                # [(bs, feature_i_dim)] * feature_dim -> (feature_dim, feature_dim, bs, out_dim)
                self.full_state_feature = full_state_feature = self.extract_state_feature(full_feature)

            # 2. extract global feature by element-wise max
            # (feature_dim, feature_dim + 1, bs, out_dim)
            full_sa_feature = torch.cat([full_state_feature, action_feature], dim=1)
            full_sa_feature, full_sa_indices = full_sa_feature.max(dim=1)  # (feature_dim, bs, out_dim)

            # 3. predict the distribution of next time step value
            full_dist = self.predict_from_sa_feature(full_sa_feature,
                                                     full_feature)  # [(bs, feature_i_dim)] * feature_dim

        if forward_masked:
            # 1. extract features of all state variables
            # (feature_dim, feature_dim, bs, out_dim)
            # masked_state_feature = self.extract_masked_state_feature(masked_feature, full_state_feature)
            masked_state_feature = self.extract_state_feature(masked_feature)

            # 2. extract global feature by element-wise max
            # mask out unused features
            # (feature_dim, feature_dim + 1, bs, out_dim)
            masked_sa_feature = torch.cat([masked_state_feature, action_feature], dim=1)
            mask = mask.permute(1, 2, 0)  # (feature_dim, feature_dim + 1, bs)
            masked_sa_feature[~mask] = float('-inf')
            masked_sa_feature, masked_sa_indices = masked_sa_feature.max(dim=1)  # (feature_dim, bs, out_dim)

            # 3. predict the distribution of next time step value
            masked_dist = self.predict_from_sa_feature(masked_sa_feature,
                                                       masked_feature)  # [(bs, feature_i_dim)] * feature_dim

        if forward_causal:
            # 1. extract features of all state variables
            causal_state_feature = self.extract_state_feature(causal_feature)

            # 2. extract global feature by element-wise max
            # mask out unused features
            # (feature_dim, feature_dim + 1, bs, out_dim)
            causal_sa_feature = torch.cat([causal_state_feature, action_feature], dim=1)
            eval_mask = self.mask.detach()  # (feature_dim, feature_dim + 1)
            causal_sa_feature[~eval_mask] = float('-inf')  # mask out non-parent features
            causal_sa_feature, causal_sa_indices = causal_sa_feature.max(dim=1)  # (feature_dim, bs, out_dim)

            # 3. predict the distribution of next time step value
            causal_dist = self.predict_from_sa_feature(causal_sa_feature,
                                                       causal_feature)  # [(bs, feature_i_dim)] * feature_dim

        return full_dist, masked_dist, causal_dist

    def extract_action_feature_abstraction(self, action):
        """
        :param action: (bs, action_dim). notice that bs must be 1D
        :return: {action_children_idx: (1, bs, out_dim)}
        """
        num_action_children = len(self.action_children_idxes)
        action = action.unsqueeze(dim=0)  # (1, bs, action_dim)
        action = action.expand(num_action_children, -1, -1)  # (num_action_children, bs, action_dim)
        # (num_action_children, bs, out_dim)
        action_feature = forward_network(action,
                                         self.abstraction_action_feature_weights,
                                         self.abstraction_action_feature_biases,
                                         dropout=self.dropout)
        action_feature = action_feature.unsqueeze(dim=1)  # (num_action_children, 1, bs, out_dim)
        action_feature = torch.unbind(action_feature, dim=0)  # [(1, bs, out_dim)] * num_action_children
        action_feature_dict = {idx: action_feature_i
                               for idx, action_feature_i in zip(self.action_children_idxes, action_feature)}

        return action_feature_dict  # {action_children_idx: (1, bs, out_dim)}

    def extract_state_feature_abstraction(self, feature):
        """
        :param feature:
            if state space is continuous: (bs, abstraction_feature_dim).
            else: [(bs, feature_i_dim)] * abstraction_feature_dim
            notice that bs must be 1D
        :return: {state_variable_idx: (num_parent, bs, out_dim)}
        """
        if self.continuous_state:
            feature = feature.transpose(0, 1)  # (abstraction_feature_dim, bs)

        features = []
        for idx, parent_idxes in self.abstraction_adjacency.items():
            feature_idx = [self.abstraction_idxes.index(parent_idx) for parent_idx in parent_idxes]
            if self.continuous_state:
                x = feature[feature_idx]  # (num_parent, bs)
                x = x.unsqueeze(dim=-1)  # (num_parent, bs, 1)
                features.append(x)
            else:
                x = [feature[parent_idx] for parent_idx in feature_idx]  # [(bs, feature_i_dim)] * num_parent
                x = [x_i.unsqueeze(dim=0) for x_i in x]  # [(1, bs, feature_i_dim)] * num_parent
                state_feature_1st_layer_weights = self.abstraction_state_feature_1st_layer_weights[idx]
                state_feature_1st_layer_biases = self.abstraction_state_feature_1st_layer_biases[idx]
                x = forward_network_batch(x,
                                          state_feature_1st_layer_weights,
                                          state_feature_1st_layer_biases,
                                          dropout=self.dropout)  # [(1, bs, out_dim)] * num_parent
                features.extend(x)
        features = torch.cat(features, dim=0)  # (total_num_parent, bs, 1)

        state_feature = forward_network(features,
                                        self.abstraction_state_feature_weights,
                                        self.abstraction_state_feature_biases,
                                        dropout=self.dropout)

        state_feature_dict = {}
        offset = 0
        for idx, parent_idxes in self.abstraction_adjacency.items():
            num_parents = len(parent_idxes)
            state_feature_dict[idx] = state_feature[offset:offset + num_parents]  # (num_parent, bs, out_dim)
            offset += num_parents
        return state_feature_dict

    def forward_step_abstraction(self, abstraction_feature, action):
        """
        :param abstraction_feature: if state space is continuous: (bs, abstraction_feature_dim)
            Otherwise: [(bs, feature_i_dim)] * abstraction_feature_dim
        :param action: (bs, action_dim)
        :return: next step value for all state variables in the format of distribution,
            if state space is continuous: a Normal distribution of shape (bs, abstraction_feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * abstraction_feature_dim,
                each of shape (bs, feature_i_dim)
        """

        # 1. extract features of all state variables and the action
        # {action_children_idx: (1, bs, out_dim)}
        action_feature = self.extract_action_feature_abstraction(action)
        # {state_variable_idx: (num_parent, bs, out_dim)}
        state_feature = self.extract_state_feature_abstraction(abstraction_feature)

        # 2. extract global feature by element-wise max
        sa_feature = []
        for idx in self.abstraction_idxes:
            sa_feature_i = state_feature[idx]
            if idx in action_feature:
                action_feature_i = action_feature[idx]  # (1, bs, out_dim)
                sa_feature_i = torch.cat([sa_feature_i, action_feature_i], dim=0)  # (num_parent + 1, bs, out_dim)
            sa_feature_i, _ = sa_feature_i.max(dim=0)  # (bs, out_dim)}
            sa_feature.append(sa_feature_i)
        # (abstraction_feature_dim, bs, out_dim)
        sa_feature = torch.stack(sa_feature, dim=0)

        # 3. predict the distribution of next time step value
        dist = self.predict_from_sa_feature(sa_feature, abstraction_feature, abstraction_mode=True)

        return dist

    def forward_with_feature(self, x, z, u, mask=None,
                             forward_mode=("full", "masked", "causal"), abstraction_mode=False):
        """
        :param x: (bs, seq_len, num_observables * num_colors)
        # :param x: [[(bs, num_colors)] * num_observables] * seq_len
        :param z: (bs, seq_len, num_colors)
        :param u:  (bs, seq_len, num_observables)
        # :param u: [(bs, num_observables)] * seq_len
        :param mask: (bs, seq_len, feature_dim, feature_dim + 1),
            randomly generated training mask used when forwarding masked_feature
        :param forward_mode
        :param abstraction_mode: whether to only forward controllable & action-relevant state variables,
            used for model-based roll-out
        :return: x_dists: [[OneHotCategorical(bs, seq_len - 2, num_colors)] * num_observables] * len(forward_mode)
                 z_dists: [[(bs, seq_len - 2, num_colors)] * 1] * len(forward_mode)
        """
        # For MLP encoder
        if u.shape[1] == 1:
            # (bs, 1, num_objects * num_colors)
            feature = torch.cat((x[:, :, :self.hidden_objects_ind[0] * self.num_colors],
                                 z, x[:, :, self.hidden_objects_ind[0] * self.num_colors:]), dim=-1)
            # [(bs, num_objects * num_colors)] * 1
            feature = torch.unbind(feature, dim=1)
            # [[(bs, num_colors)] * num_objects] * 1
            feature = [torch.split(feature_i, self.num_colors, dim=-1) for feature_i in feature]
            # [(bs, num_observables)] * 1
            action = torch.unbind(u, dim=1)
        else:
            # (bs, seq_len - 1, num_objects * num_colors)
            feature = torch.cat((x[:, :-1, :self.hidden_objects_ind[0] * self.num_colors],
                                 z[:, 1:], x[:, :-1, self.hidden_objects_ind[0] * self.num_colors:]), dim=-1)
            # [(bs, num_objects * num_colors)] * (seq_len - 1)
            feature = torch.unbind(feature, dim=1)
            # states from t=1 to T-1
            # [[(bs, num_colors)] * num_objects] * (seq_len - 1)
            feature = [torch.split(feature_i, self.num_colors, dim=-1) for feature_i in feature]
            # [(bs, num_observables)] * (seq_len - 1)
            action = torch.unbind(u[:, 1:], dim=1)

        # # For masked MLP encoder
        # x = x[:-1]  # [[(bs, num_colors)] * num_observables] * (seq_len - 1)
        # z = z[:, 1:]
        # z = torch.unbind(z, dim=1)  # [(bs, num_colors)] * (seq_len - 1)
        # # states from t=1 to T-1
        # # [[(bs, num_colors)] * num_objects] * (seq_len - 1)
        # feature = [x_t[:self.hidden_objects_ind[0]] + [z_t] + x_t[self.hidden_objects_ind[-1]:]
        #            for x_t, z_t in zip(x, z)]
        # # [(bs, num_observables)] * (seq_len - 1)
        # action = u[1:]

        # [(bs, feature_dim, feature_dim + 1)] * (seq_len - 1)
        mask = torch.unbind(mask[:, 1:], dim=1) if mask is not None else [None for _ in feature]

        # full_feature: prediction using all state variables
        # masked_feature: prediction using state variables specified by mask
        # causal_feature: prediction using causal parents (inferred so far)
        full_feature = feature if "full" in forward_mode else [None for _ in feature]
        masked_feature = feature if "masked" in forward_mode else [None for _ in feature]
        causal_feature = feature if "causal" in forward_mode else [None for _ in feature]

        if abstraction_mode:
            assert not self.use_cache
            forward_mode = ("causal",)
            full_feature = masked_feature = None
            if self.abstraction_quested:
                if self.continuous_state:
                    causal_feature = causal_feature[:, self.abstraction_idxes]
                else:
                    causal_feature = [causal_feature[idx] for idx in self.abstraction_idxes]

        modes = ["full", "masked", "causal"]
        assert all([ele in modes for ele in forward_mode])
        if "masked" in forward_mode:
            assert mask is not None

        full_dists, masked_dists, causal_dists = [], [], []
        sa_feature_cache = []

        # forward prediction of next state from t=1 to T-1
        for i in range(len(action)):
            if self.use_cache and self.sa_feature_cache:
                # only used when evaluate with the same state and action a lot in self.update_mask()
                action_feature, full_state_feature = self.sa_feature_cache[i]
            else:
                action_feature, full_state_feature = None, None

            full_dist = masked_dist = None
            if abstraction_mode and self.abstraction_quested:
                causal_dist = self.forward_step_abstraction(causal_feature, action)
            else:
                full_dist, masked_dist, causal_dist = \
                    self.forward_step(full_feature[i], masked_feature[i], causal_feature[i], action[i],
                                      mask[i], action_feature, full_state_feature)  # one-step forward prediction
            full_dists.append(full_dist)
            masked_dists.append(masked_dist)
            causal_dists.append(causal_dist)
            sa_feature_cache.append((self.action_feature, self.full_state_feature))

        if self.use_cache and self.sa_feature_cache is None:
            self.sa_feature_cache = sa_feature_cache

        # full/masked/causal_dists: [[OneHotCategorical(bs, num_colors)] * num_objects] * (seq_len - 1)
        dists = [full_dists, masked_dists, causal_dists]
        x_dists, z_dists = [], []
        for mode in forward_mode:
            dist = dists[modes.index(mode)]
            # t=2 to T
            # [OneHotCategorical(bs, seq_len - 1, num_colors)] * num_observables
            dist = self.stack_dist(dist)
            # x_dists.append([OneHotCategorical(probs=dist[i].probs[:, :-1]) for i in self.obs_objects_ind])
            # # x_dists.append([dist[i] for i in self.obs_objects_ind])
            # z_dists.append([dist[i].probs[:, :-1] for i in self.hidden_objects_ind])
            x_dists.append([dist[i] for i in self.obs_objects_ind])
            z_dists.append([dist[i].probs for i in self.hidden_objects_ind])

        if len(forward_mode) == 1:
            return x_dists[0], z_dists[0]

        # t=2 to T
        # x_dists: [[OneHotCategorical(bs, seq_len - 1, num_colors)] * num_observables] * len(forward_mode)
        # z_dists: [[(bs, seq_len - 1, num_colors)] * num_hiddens] * len(forward_mode)
        return x_dists, z_dists

    def restore_batch_size_shape(self, dist, bs):
        # restore multi-dimensional batch size
        if self.continuous_state:
            mu, std = dist.mean, dist.stddev  # (bs, n_pred_step, feature_dim)
            mu = mu.view(*bs, *mu.shape[-2:])  # (*bs, n_pred_step, feature_dim)
            std = std.view(*bs, *std.shape[-2:])  # (*bs, n_pred_step, feature_dim)
            return Normal(mu, std)
        else:
            # [(bs, n_pred_step, feature_i_dim)] * feature_dim
            dist_list = []
            for dist_i in dist:
                if isinstance(dist_i, Normal):
                    mu, std = dist.mean, dist.stddev  # (bs, n_pred_step, feature_i_dim)
                    mu = mu.view(*bs, *mu.shape[-2:])  # (*bs, n_pred_step, feature_i_dim)
                    std = std.view(*bs, *std.shape[-2:])  # (*bs, n_pred_step, feature_i_dim)
                    dist_i = Normal(mu, std)
                elif isinstance(dist_i, OneHotCategorical):
                    logits = dist_i.logits  # (bs, n_pred_step, feature_i_dim)
                    logits = logits.view(*bs, *logits.shape[-2:])  # (*bs, n_pred_step, feature_i_dim)
                    dist_i = OneHotCategorical(logits=logits)
                else:
                    raise NotImplementedError
                dist_list.append(dist_i)

            return dist_list

    def forward(self, obs, actions, mask=None, forward_mode=("full", "masked", "causal"),
                abstraction_mode=False):
        feature = self.get_feature(obs)
        return self.forward_with_feature(feature, actions, mask, forward_mode, abstraction_mode)

    def setup_annealing(self, step):
        super(InferenceCMI, self).setup_annealing(step)

    def get_mask_by_id(self, mask_ids):
        """
        :param mask_ids: (bs_1, bs_2, ..., bs_n, feature_dim), idxes of state variable to drop
        :return: (bs_1, bs_2, ..., bs_n, feature_dim, feature_dim + 1), bool mask of state variables to use
        """
        int_mask = F.one_hot(mask_ids, self.feature_dim + 1)
        bool_mask = int_mask < 1
        return bool_mask

    def get_training_mask(self, batch_size, seq_len):
        # uniformly select one state variable to omit when predicting the next time step value
        idxes = torch.randint(self.feature_dim + 1, (batch_size, seq_len, self.feature_dim))
        return self.get_mask_by_id(idxes)  # (bs, seq_len, feature_dim, feature_dim + 1)

    def get_eval_mask(self, batch_size, seq_len, i):
        # omit i-th state variable or the action when predicting the next time step value
        feature_dim = self.feature_dim
        idxes = torch.full((batch_size, seq_len, feature_dim), fill_value=i,
                           dtype=torch.int64, device=self.device)
        # self_mask = torch.arange(feature_dim, device=self.device)
        # each state variable must depend on itself when predicting the next time step value
        # idxes[idxes >= self_mask] += 1
        return self.get_mask_by_id(idxes)  # (bs, seq_len, feature_dim, feature_dim + 1)

    def get_enc_mask(self, bs, i, feature_dim_dset):
        # omit i-th next observable when encoding the current hidden
        num_hiddens = len(self.hidden_objects_ind)
        mask_ids = torch.full((bs, num_hiddens), i, dtype=torch.int64, device=self.device)
        int_mask = F.one_hot(mask_ids, feature_dim_dset + 1)
        bool_mask = int_mask < 1
        return bool_mask  # (bs, num_hidden_objects, feature_dim_dset + 1)

    def prediction_loss_from_multi_dist(self, pred_next_dist, next_feature, loss_type="recon"):
        """
        calculate total prediction loss for full, masked and/or causal prediction distributions
        :param pred_next_dist:
            a list of prediction distributions under different prediction mode
            if state space is continuous:
                a Normal distribution of shape (bs, n_pred_step, feature_dim)
            else:
                a list of distributions, [OneHotCategorical / Normal] * feature_dim,
                each of shape (bs, n_pred_step, feature_i_dim)
        :param next_feature:
            if state space is continuous:
                a tensor of shape (bs, n_pred_step, feature_dim)
            else:
                a list of tensors, [(bs, n_pred_step, feature_i_dim)] * feature_dim
        :param loss_type: "recon" or "kl"
        :return pred_loss: scalar tensor
                pred_loss_detail: {"loss_name": loss_value}
        """
        pred_losses = [self.prediction_loss_from_dist(pred_next_dist_i, next_feature, loss_type=loss_type)
                       for pred_next_dist_i in pred_next_dist]  # [(bs, n_pred_step)] * len(forward_mode)

        if len(pred_losses) == 2:
            pred_losses.append(None)
        assert len(pred_losses) == 3
        full_pred_loss, masked_pred_loss, causal_pred_loss = pred_losses

        full_pred_loss = full_pred_loss.sum(dim=-1).mean()  # sum over n_pred_step then average over bs
        masked_pred_loss = masked_pred_loss.sum(dim=-1).mean()
        # full_pred_loss = full_pred_loss[:, 0].mean()  # slice the first pred step then average over bs
        # masked_pred_loss = masked_pred_loss[:, 0].mean()

        pred_loss = full_pred_loss + masked_pred_loss

        if loss_type == "recon":
            pred_loss_detail = {"full_nll_loss": full_pred_loss,
                                "masked_nll_loss": masked_pred_loss}
        elif loss_type == "kl":
            pred_loss_detail = {"full_kl_loss": full_pred_loss,
                                "masked_kl_loss": masked_pred_loss}
        elif loss_type == "recon_h":
            pred_loss_detail = {"full_nll_h_loss": full_pred_loss,
                                "masked_nll_h_loss": masked_pred_loss}
        else:
            raise NotImplementedError

        if causal_pred_loss is not None:
            causal_pred_loss = causal_pred_loss.sum(dim=-1).mean()
            # causal_pred_loss = causal_pred_loss[:, 0].mean()
            pred_loss += causal_pred_loss
            if loss_type == "recon":
                pred_loss_detail["causal_nll_loss"] = causal_pred_loss
            elif loss_type == "kl":
                pred_loss_detail["causal_kl_loss"] = causal_pred_loss
            elif loss_type == "recon_h":
                pred_loss_detail["causal_nll_h_loss"] = causal_pred_loss
            else:
                raise NotImplementedError

        return pred_loss, pred_loss_detail

    def rew_loss_from_feature(self, feature, rew):
        """
        calculate predicted reward losses from full features
        :param feature: [(bs, n_pred_step, num_colors)] *  (2 * num_objects)
        :param rew: (bs, n_pred_step, 1)
        :return rew_loss: scalar tensor
                rew_loss_detail: {"loss_name": loss_value}
        """
        # (bs, n_pred_step, reward_dim)
        rew_feature = self.decoder(feature)

        if self.decoder.categorical_rew:
            # (bs, n_pred_step)
            rew_unnorm = (rew * self.feature_dim).squeeze(dim=-1).long()
            # (bs, reward_dim, n_pred_step)
            rew_feature = rew_feature.permute(0, 2, 1)
            if not self.training:
                # accuracy as the prediction loss
                _, rew_feature_ids = rew_feature.max(dim=1)  # (bs, n_pred_step)
                rec_loss = (rew_feature_ids == rew_unnorm).sum() / np.prod(rew_feature_ids.shape)
            else:
                rec_loss = self.loss_ce(rew_feature, rew_unnorm)  # (bs, n_pred_step)
        else:
            # L1/L2 reward prediction loss
            rec_loss = self.loss_l1(rew_feature, rew)  # (bs, n_pred_step, 1)
            rec_loss = rec_loss.squeeze(-1)

        rew_loss = rec_loss.sum(dim=-1).mean()  # sum over n_pred_step then average over bs
        rew_loss_detail = {"rew_loss": rew_loss}

        return rew_loss, rew_loss_detail

    # def backprop(self, loss, loss_detail):
    #     encoder_freq = self.encoder_params.encoder_freq
    #     transition_freq = self.inference_params.transition_freq
    #     alter_steps = self.inference_params.alter_steps
    #     freq = encoder_freq + transition_freq
    #     if self.update_num <= alter_steps:
    #         if self.update_num % freq < encoder_freq:
    #             optimizer = self.optimizer_encoder
    #         else:
    #             optimizer = self.optimizer_transition
    #     else:
    #         optimizer = self.optimizer
    #     self.optimizer_encoder.zero_grad()
    #     self.optimizer_transition.zero_grad()
    #     optimizer.zero_grad()
    #     loss.backward()
    #
    #     grad_clip_norm = self.inference_params.grad_clip_norm
    #     if not grad_clip_norm:
    #         grad_clip_norm = np.inf
    #     loss_detail["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
    #
    #     optimizer.step()
    #     return loss_detail

    def backprop(self, nelbo, rew_loss, loss_detail):
        pre_train_steps = self.inference_params.pre_train_steps
        encoder_freq = self.encoder_params.encoder_freq
        transition_freq = self.inference_params.transition_freq
        alter_steps = self.inference_params.alter_steps
        freq = encoder_freq + transition_freq
        if self.update_num <= pre_train_steps:
            loss = rew_loss
            optimizer = self.optimizer_enc_decoder
        else:
            loss = nelbo + 20 * rew_loss
            # loss = nelbo
            if (self.update_num - pre_train_steps) <= alter_steps:
                if (self.update_num - pre_train_steps) % freq < transition_freq:
                    optimizer = self.optimizer_transition
                else:
                    optimizer = self.optimizer_enc_decoder
            else:
                optimizer = self.optimizer

        self.optimizer_enc_decoder.zero_grad()
        self.optimizer_transition.zero_grad()
        optimizer.zero_grad()
        loss.backward()

        grad_clip_norm = self.inference_params.grad_clip_norm
        if not grad_clip_norm:
            grad_clip_norm = np.inf
        loss_detail["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)

        optimizer.step()
        return loss_detail

    # def update(self, obs, hidden, eval=False):
    #     """
    #     KL for hiddens
    #     :param obs: Batch(obs_i_key: (bs, stack_num, obs_i_shape))
    #     :param hidden: Batch(hidden_i_key: (bs, stack_num, 1))
    #     :return: {"loss_name": loss_value}
    #     """
    #     self.update_num += 1
    #
    #     eval_freq = self.cmi_params.eval_freq
    #     inference_gradient_steps = self.params.training_params.inference_gradient_steps
    #     forward_mode = ("full", "masked", "causal")
    #
    #     z, z_probs, x, u, st, r = self.encoder(obs)
    #     bs, seq_len = z.shape[:2]
    #     mask = self.get_training_mask(bs, seq_len)  # (bs, seq_len, feature_dim, feature_dim + 1)
    #     # predicted next observables from t=2 to T
    #     # predicted next hidden from t=2 to T-1
    #     # x_dists: [[OneHotCategorical(bs, seq_len - 1, num_colors)] * num_observables] * len(forward_mode)
    #     # z_dists: [[(bs, seq_len - 2, num_colors)] * num_hiddens] * len(forward_mode)
    #     next_x_dists, next_z_prior_dists = self.forward_with_feature(x, z, u, mask, forward_mode=forward_mode)
    #
    #     # For MLP encoder
    #     # true next observables from t=2 to T
    #     # [(bs, seq_len - 1, num_colors)] * num_observables
    #     next_x = torch.split(x[:, 1:], self.num_colors, dim=-1)
    #     # inferenced next hidden from t=2 to T-1
    #     # [(bs, seq_len - 2, num_colors)] * num_hiddens
    #     next_z_infer_probs = torch.split(z_probs[:, 2:], self.num_colors, dim=-1)
    #
    #     if not self.update_num % (eval_freq * inference_gradient_steps):
    #         next_x_dists = next_x_dists[:2]
    #     recon, recon_detail = self.prediction_loss_from_multi_dist(next_x_dists, next_x, loss_type="recon")
    #     kl, kl_detail = self.prediction_loss_from_multi_dist(next_z_prior_dists, next_z_infer_probs, loss_type="kl")
    #     rew_loss, rew_loss_detail = self.rew_loss_from_feature(st, r)
    #
    #     # loss = recon + kl + rew_loss
    #     nelbo = recon + kl
    #     loss_detail = {**recon_detail, **kl_detail, **rew_loss_detail}
    #     # loss_detail = {**recon_detail, **kl_detail}
    #
    #     # if not eval and torch.isfinite(nelbo):
    #     #     self.backprop(nelbo, loss_detail)
    #
    #     if not eval and torch.isfinite(nelbo):
    #         self.backprop(nelbo, rew_loss, loss_detail)
    #
    #     return loss_detail

    def update(self, obs, hidden, eval=False):
        """
        NLL for hiddens
        :param obs: Batch(obs_i_key: (bs, stack_num, obs_i_shape))
        :param hidden: Batch(hidden_i_key: (bs, stack_num, 1))
        :return: {"loss_name": loss_value}
        """
        self.update_num += 1

        eval_freq = self.cmi_params.eval_freq
        inference_gradient_steps = self.params.training_params.inference_gradient_steps
        forward_mode = ("full", "masked", "causal")

        z, z_probs, x, u, st, r = self.encoder(obs, self.forward_with_feature)
        # z, z_probs, x, u, st, r = self.encoder(obs)
        bs, seq_len = z.shape[:2]
        mask = self.get_training_mask(bs, seq_len)  # (bs, seq_len, feature_dim, feature_dim + 1)
        # predicted next observables from t=2 to T
        # predicted next hidden from t=2 to T
        # x_dists: [[OneHotCategorical(bs, seq_len - 1, num_colors)] * num_observables] * len(forward_mode)
        # z_dists: [[(bs, seq_len - 1, num_colors)] * num_hiddens] * len(forward_mode)
        next_x_dists, next_z_prior_dists = self.forward_with_feature(x, z, u, mask, forward_mode=forward_mode)
        # t=2 to T-1
        # [[OneHotCategorical(bs, seq_len - 2, num_colors)] * num_observables] * len(forward_mode)
        next_x_dists = [[OneHotCategorical(probs=dist_i.probs[:, :-1]) for dist_i in forward_mode_i]
                        for forward_mode_i in next_x_dists]
        # # [[OneHotCategorical(bs, seq_len - 3, num_colors)] * num_hiddens] * len(forward_mode)
        # next_z_prior_dists = [[OneHotCategorical(probs=dist_i[:, :-1]) for dist_i in forward_mode_i]
        #                       for forward_mode_i in next_z_prior_dists]
        # [[(bs, seq_len - 2, num_colors)] * num_hiddens] * len(forward_mode)
        next_z_prior_dists = [[dist_i[:, :-1] for dist_i in forward_mode_i]
                              for forward_mode_i in next_z_prior_dists]

        # For MLP encoder
        # true next observables at t=2 to T-1
        # [(bs, seq_len - 2, num_colors)] * num_observables
        next_x = torch.split(x[:, 1:-1], self.num_colors, dim=-1)
        # inferenced next hidden at t=2 to T-1
        # [(bs, seq_len - 2, num_colors)] * num_hiddens
        # next_z_infer_probs = torch.split(z[:, 2:-1], self.num_colors, dim=-1)
        next_z_infer_probs = torch.split(z_probs[:, 2:], self.num_colors, dim=-1)

        # # For masked MLP encoder
        # # true next observables from t=2 to T-1
        # # [[(bs, num_colors)] * num_observables] * (seq_len - 2)
        # next_x = x[1:-1]
        # # [(bs, seq_len - 2, num_colors)] * num_observables
        # next_x = self.stack_dist(next_x)
        # # inferenced next hidden from t=2 to T-1
        # # [(bs, seq_len - 2, num_colors)] * 1
        # next_z_infer_probs = [z[:, 2:]]
        # # next_feature = (next_x[:self.hidden_objects_ind[0]] + next_z_infer_probs
        # #                 + next_x[self.hidden_objects_ind[-1]:])

        if not self.update_num % (eval_freq * inference_gradient_steps):
            next_x_dists = next_x_dists[:2]
        recon, recon_detail = self.prediction_loss_from_multi_dist(next_x_dists, next_x, loss_type="recon")
        recon_h, recon_h_detail = self.prediction_loss_from_multi_dist(next_z_prior_dists, next_z_infer_probs,
                                                                       loss_type="kl")
        # recon, recon_detail = self.prediction_loss_from_multi_dist(next_dists, next_feature, loss_type="recon")
        rew_loss, rew_loss_detail = self.rew_loss_from_feature(st, r)
        # rew_loss, rew_loss_detail = torch.tensor(0.), {"rew_loss": torch.tensor(0.)}

        # # hidden from t=1 to T-1
        # # (bs, seq_len - 1)
        # hidden_label = hidden['obj1'][:, :-1].squeeze(-1).long()
        # # (bs, num_colors, seq_len - 1)
        # hidden_logits = z_logits[:, 1:].permute(0, 2, 1)
        # # (bs, seq_len - 1)
        # hidden_loss = self.loss_ce(hidden_logits, hidden_label)
        # hidden_loss = hidden_loss.sum(dim=-1).mean()  # sum over n_pred_step then average over bs
        # hidden_loss_detail = {"hidden_loss": hidden_loss}

        # loss = recon + kl + rew_loss
        nelbo = recon + recon_h
        # loss_detail = {**recon_detail, **recon_h_detail, **rew_loss_detail, **hidden_loss_detail}
        loss_detail = {**recon_detail, **recon_h_detail, **rew_loss_detail}

        # if not eval and torch.isfinite(nelbo):
        #     self.backprop(nelbo, loss_detail)

        if not eval and torch.isfinite(nelbo):
            self.backprop(nelbo, rew_loss, loss_detail)

        return loss_detail

    # def update_mask(self, obs, hidden=None):
    #     """
    #     KL for hiddens
    #     1. use estimated full_dists and masked_dists to compute the CMI matrix 'eval_step_CMI'
    #     for all feature pairs
    #     2. exponentially smooth 'self.mask_CMI' given the new-coming 'eval_step_CMI'
    #     3. threshold 'self.mask_CMI' to get the boolean adjacency matrix 'self.mask'
    #     with shape (feature_dim, feature_dim + 1)
    #     :param obs: Batch(obs_i_key: (bs, stack_num, obs_i_shape))
    #     :param hidden: Batch(hidden_i_key: (bs, stack_num, hidden_i_shape))
    #     :return: {"loss_name": loss_value}
    #     """
    #     feature_dim = self.feature_dim
    #
    #     # set up cache for faster computation
    #     self.use_cache = False
    #     self.sa_feature_cache = None
    #
    #     eval_details = {}
    #     next_z_prior_masked_ = []
    #     masked_recon_ = []
    #     with torch.no_grad():
    #         z, z_probs, x, u, st, r = self.encoder(obs)
    #         bs, seq_len = z.shape[:2]
    #
    #         # For MLP encoder
    #         # true next observables from t=2 to T
    #         # [(bs, seq_len - 1, num_colors)] * num_observables
    #         next_x = torch.split(x[:, 1:], self.num_colors, dim=-1)
    #         # inferenced next hidden from t=2 to T-1
    #         # [(bs, seq_len - 2, num_colors)] * num_hiddens
    #         next_z_infer_feature = torch.split(z[:, 2:], self.num_colors, dim=-1)
    #
    #         for i in range(feature_dim + 1):
    #             mask = self.get_eval_mask(bs, seq_len, i)                         # (bs, feature_dim, feature_dim + 1)
    #             if i == 0:
    #                 # predicted next observables from t=2 to T
    #                 # predicted next hidden from t=2 to T-1
    #                 # x_dists: [[OneHotCategorical(bs, seq_len - 1, num_colors)] * num_observables] * len(forward_mode)
    #                 # z_dists: [[(bs, seq_len - 2, num_colors)] * num_hidden] * len(forward_mode)
    #                 next_x_dists, next_z_prior_dists = self.forward_with_feature(x, z, u, mask)
    #                 # (bs, seq_len - 2, num_hidden, num_colors)
    #                 next_z_prior_full = torch.stack(next_z_prior_dists[0], dim=-2)
    #                 print("next_z_prior_full", next_z_prior_full[:2])
    #                 next_z_prior_masked = torch.stack(next_z_prior_dists[1], dim=-2)
    #
    #                 # (bs, seq_len, 1, num_colors)
    #                 z_infer_probs = z_probs[:, :].unsqueeze(dim=-2)
    #                 print("z_infer_probs", z_infer_probs[:2])
    #
    #                 # recon: (bs, seq_len - 1, num_observables)
    #                 full_recon, masked_recon, eval_recon = \
    #                     [self.prediction_loss_from_dist(next_x_dists_i, next_x, keep_variable_dim=True,
    #                                                     loss_type="recon")
    #                      for next_x_dists_i in next_x_dists]
    #
    #                 if hidden is not None:
    #                     # hidden objects at t=T-1
    #                     # (bs, num_hiddens)
    #                     next_hidden = [hidden[key][:, -2].squeeze(dim=-1).long()
    #                                    for key in self.params.hidden_keys]
    #                     next_hidden = torch.stack(next_hidden, dim=-1)
    #                     next_feature_hidden = [z_i[:, -1].argmax(-1) for z_i in next_z_infer_feature]
    #                     next_feature_hidden = torch.stack(next_feature_hidden, dim=-1)
    #                     pred_next_feature_hidden = [z_i[:, -1].argmax(-1)
    #                                                 for z_i in next_z_prior_dists[0]]
    #                     pred_next_feature_hidden = torch.stack(pred_next_feature_hidden, dim=-1)
    #
    #                     # hidden prediction accuracy
    #                     # (num_hiddens, )
    #                     next_enc_hidden_acc = (next_feature_hidden == next_hidden).sum(0) / bs
    #                     next_pred_hidden_acc = (pred_next_feature_hidden == next_hidden).sum(0) / bs
    #                     next_pred_enco_hidden_acc = (pred_next_feature_hidden == next_feature_hidden).sum(0) / bs
    #
    #                     print("next_true_hidden", next_hidden[:20].t())
    #                     print("next_enco_hidden", next_feature_hidden[:20].t())
    #                     print("next_pred_hidden", pred_next_feature_hidden[:20].t())
    #                     eval_details["next_enc_hidden_acc"] = {"h_{}".format(self.hidden_objects_ind[0]):
    #                                                            next_enc_hidden_acc[0]}
    #                     eval_details["next_pred_hidden_acc"] = {"h_{}".format(self.hidden_objects_ind[0]):
    #                                                             next_pred_hidden_acc[0]}
    #                     eval_details["next_pred_enco_hidden_acc"] = {"h_{}".format(self.hidden_objects_ind[0]):
    #                                                                  next_pred_enco_hidden_acc[0]}
    #             else:
    #                 # x_dists: [OneHotCategorical(bs, seq_len - 1, num_colors)] * num_observables
    #                 # z_dists: [(bs, seq_len - 2, num_colors)] * 1
    #                 next_x_dists, next_z_prior_dists = self.forward_with_feature(x, z, u, mask,
    #                                                                              forward_mode=("masked",))
    #                 # (bs, seq_len - 2, num_hidden, num_colors)
    #                 next_z_prior_masked = torch.stack(next_z_prior_dists, dim=-2)
    #                 # (bs, seq_len - 1, num_observables)
    #                 masked_recon = self.prediction_loss_from_dist(next_x_dists, next_x, keep_variable_dim=True,
    #                                                               loss_type="recon")
    #             next_z_prior_masked_.append(next_z_prior_masked)
    #             masked_recon_.append(masked_recon)
    #         eval_recon = eval_recon.sum(dim=-1).mean()    # scalar
    #         eval_details["eval_recon"] = eval_recon
    #         rew_loss, rew_loss_detail = self.rew_loss_from_feature(st, r)
    #
    #     # clean cache
    #     self.use_cache = False
    #     self.sa_feature_cache = None
    #     self.action_feature = None
    #     self.full_state_feature = None
    #
    #     # log(T_o(full) / T_o(masked))
    #     # (bs, seq_len - 1, num_observables, 1)
    #     full_recon = full_recon.unsqueeze(-1)
    #     # (bs, seq_len - 1, num_observables, feature_dim + 1)
    #     masked_recon_ = torch.stack(masked_recon_, dim=-1)
    #     CMI_o = masked_recon_ - full_recon
    #     CMI_o = CMI_o[:, 0]
    #     # (num_observables, feature_dim + 1)
    #     # CMI_o = CMI_o.mean(dim=(0, 1))
    #     CMI_o = CMI_o.mean(dim=0)
    #
    #     # KL(T_h(full) || T_h(masked))
    #     # (bs, seq_len - 2, num_hidden, 1, num_colors)
    #     next_z_prior_full = next_z_prior_full.unsqueeze(-2)
    #     # (bs, seq_len - 2, num_hidden, feature_dim + 1, num_colors)
    #     next_z_prior_masked_ = torch.stack(next_z_prior_masked_, dim=-2)
    #     # summed over num_colors for KL as CMI of the hidden
    #     # (bs, seq_len - 2, num_hidden, feature_dim + 1)
    #     CMI_h = torch.sum(next_z_prior_full * (torch.log(next_z_prior_full + 1e-20) -
    #                                            torch.log(next_z_prior_masked_ + 1e-20)), dim=-1)
    #     CMI_h = CMI_h[:, 0]
    #     # (num_hiddens, feature_dim + 1)
    #     # CMI_h = CMI_h.mean(dim=(0, 1))
    #     CMI_h = CMI_h.mean(dim=0)
    #
    #     CMI = torch.cat((CMI_o[:self.hidden_objects_ind[0]], CMI_h, CMI_o[self.hidden_objects_ind[0]:]),
    #                     dim=0)  # (feature_dim, feature_dim + 1)
    #
    #     self.eval_step_CMI += CMI
    #     self.mask_update_idx += 1
    #
    #     eval_steps = self.cmi_params.eval_steps
    #     eval_tau = self.cmi_params.eval_tau
    #     if self.mask_update_idx == eval_steps:
    #         self.eval_step_CMI /= eval_steps
    #         self.mask_CMI = self.mask_CMI * eval_tau + self.eval_step_CMI * (1 - eval_tau)  # Exponential smoothing
    #
    #         # ensure each object has at least one parent
    #         mask_high_thres = self.mask_CMI >= self.CMI_threshold
    #         mask_CMI_max, _ = self.mask_CMI.max(dim=1, keepdim=True)
    #         mask_low_thres = (self.mask_CMI < self.CMI_threshold) * (self.mask_CMI == mask_CMI_max)
    #         self.mask = mask_high_thres + mask_low_thres
    #     loss_details = {**eval_details, **rew_loss_detail}
    #     return loss_details

    def update_mask(self, obs, hidden=None):
        """
        NLL for hiddens
        1. use estimated full_dists and masked_dists to compute the CMI matrix 'eval_step_CMI'
        for all feature pairs
        2. exponentially smooth 'self.mask_CMI' given the new-coming 'eval_step_CMI'
        3. threshold 'self.mask_CMI' to get the boolean adjacency matrix 'self.mask'
        with shape (feature_dim, feature_dim + 1)
        :param obs: Batch(obs_i_key: (bs, stack_num, obs_i_shape))
        :param hidden: Batch(hidden_i_key: (bs, stack_num, hidden_i_shape))
        :return: {"loss_name": loss_value}
        """
        feature_dim = self.feature_dim

        self.update_num_eval += 1

        # set up cache for faster computation
        self.use_cache = False
        self.sa_feature_cache = None

        eval_details = {}
        next_z_prior_masked_ = []
        masked_recon_ = []
        masked_recon_h_ = []
        with torch.no_grad():
            z, z_probs, x, u, st, r = self.encoder(obs, self.forward_with_feature)
            # z, z_probs, x, u, st, r = self.encoder(obs)
            bs, seq_len = z.shape[:2]

            # For MLP encoder
            # true next observables at t=2 to T-1
            # [(bs, seq_len - 2, num_colors)] * num_observables
            next_x = torch.split(x[:, 1:-1], self.num_colors, dim=-1)
            # inferenced next hidden at t=2 to T-1
            # [(bs, seq_len - 2, num_colors)] * num_hiddens
            # next_z_infer_feature = torch.split(z[:, 2:-1], self.num_colors, dim=-1)
            next_z_infer_feature = torch.split(z_probs[:, 2:], self.num_colors, dim=-1)
            # (bs, seq_len - 2, num_hiddens, num_colors)
            z_infer_probs = torch.stack(next_z_infer_feature, dim=-2)

            # # For masked MLP encoder
            # # true next observables from t=2 to T-1
            # next_x = x[1:-1]
            # # [(bs, seq_len - 2, num_colors)] * num_observables
            # next_x = self.stack_dist(next_x)
            # # [(bs, seq_len - 2, num_colors)] * 1
            # next_z_infer_feature = [z[:, 2:]]

            for i in range(feature_dim + 1):
                mask = self.get_eval_mask(bs, seq_len, i)  # (bs, feature_dim, feature_dim + 1)
                if i == 0:
                    # predicted next observables from t=2 to T
                    # predicted next hidden from t=2 to T
                    # x_dists: [[OneHotCategorical(bs, seq_len - 1, num_colors)] * num_observables] * len(forward_mode)
                    # z_dists: [[(bs, seq_len - 1, num_colors)] * num_hidden] * len(forward_mode)
                    next_x_dists, next_z_prior_dists = self.forward_with_feature(x, z, u, mask)
                    # t=2 to T-1
                    # [[OneHotCategorical(bs, seq_len - 2, num_colors)] * num_observables] * len(forward_mode)
                    next_x_dists = [[OneHotCategorical(probs=dist_i.probs[:, :-1]) for dist_i in forward_mode_i]
                                    for forward_mode_i in next_x_dists]
                    # [[OneHotCategorical(bs, seq_len - 3, num_colors)] * num_hiddens] * len(forward_mode)
                    # next_z_prior_dists = [[OneHotCategorical(probs=dist_i[:, :-1]) for dist_i in forward_mode_i]
                    #                       for forward_mode_i in next_z_prior_dists]
                    # [[(bs, seq_len - 2, num_colors)] * num_hiddens] * len(forward_mode)
                    next_z_prior_dists = [[dist_i[:, :-1] for dist_i in forward_mode_i]
                                          for forward_mode_i in next_z_prior_dists]
                    # (bs, seq_len - 2, num_hiddens, num_colors)
                    next_z_prior_full = torch.stack(next_z_prior_dists[0], dim=-2)
                    next_z_prior_masked = torch.stack(next_z_prior_dists[1], dim=-2)

                    # recon: (bs, seq_len - 2, num_observables)
                    full_recon, masked_recon, eval_recon = \
                        [self.prediction_loss_from_dist(next_x_dists_i, next_x,
                                                        keep_variable_dim=True,
                                                        loss_type="recon")
                         for next_x_dists_i in next_x_dists]
                    # # recon: (bs, seq_len - 3, num_hiddens)
                    # full_recon_h, masked_recon_h, eval_recon_h = \
                    #     [self.prediction_loss_from_dist(next_z_dists_i, next_z_infer_feature,
                    #                                     keep_variable_dim=True,
                    #                                     loss_type="recon_h")
                    #      for next_z_dists_i in next_z_prior_dists]

                    if hidden is not None:
                        # hidden objects at t=2
                        # (bs, num_hiddens)
                        next_hidden = [hidden[key][:, 1].squeeze(dim=-1).long()
                                       for key in self.params.hidden_keys]
                        next_hidden = torch.stack(next_hidden, dim=-1)
                        next_feature_hidden = [z_i[:, -1].argmax(-1) for z_i in next_z_infer_feature]
                        next_feature_hidden = torch.stack(next_feature_hidden, dim=-1)
                        # pred_next_feature_hidden = [z_i.probs[:, -1].argmax(-1)
                        #                             for z_i in next_z_prior_dists[0]]
                        pred_next_feature_hidden = [z_i[:, -1].argmax(-1)
                                                    for z_i in next_z_prior_dists[0]]
                        pred_next_feature_hidden = torch.stack(pred_next_feature_hidden, dim=-1)

                        # hidden prediction accuracy
                        # (num_hiddens, )
                        next_enc_hidden_acc = (next_feature_hidden == next_hidden).sum(0) / bs
                        next_pred_hidden_acc = (pred_next_feature_hidden == next_hidden).sum(0) / bs
                        next_pred_enco_hidden_acc = (pred_next_feature_hidden == next_feature_hidden).sum(0) / bs

                        eval_details["next_enc_hidden_acc"] = {}
                        eval_details["next_pred_hidden_acc"] = {}
                        eval_details["next_pred_enco_hidden_acc"] = {}
                        for j in range(len(self.hidden_objects_ind)):
                            key = "h_{}".format(self.hidden_objects_ind[j])
                            eval_details["next_enc_hidden_acc"][key] = next_enc_hidden_acc[j]
                            eval_details["next_pred_hidden_acc"][key] = next_pred_hidden_acc[j]
                            eval_details["next_pred_enco_hidden_acc"][key] = next_pred_enco_hidden_acc[j]
                else:
                    # x_dists: [OneHotCategorical(bs, seq_len - 1, num_colors)] * num_observables
                    # z_dists: [(bs, seq_len - 1, num_colors)] * num_hiddens
                    next_x_dists, next_z_prior_dists = self.forward_with_feature(x, z, u, mask,
                                                                                 forward_mode=("masked",))
                    next_x_dists = [OneHotCategorical(probs=dist_i.probs[:, :-1]) for dist_i in next_x_dists]
                    # next_z_prior_dists = [OneHotCategorical(probs=dist_i[:, :-1]) for dist_i in next_z_prior_dists]
                    next_z_prior_dists = [dist_i[:, :-1] for dist_i in next_z_prior_dists]
                    # (bs, seq_len - 2, num_hiddens, num_colors)
                    next_z_prior_masked = torch.stack(next_z_prior_dists, dim=-2)

                    # (bs, seq_len - 2, num_observables)
                    masked_recon = self.prediction_loss_from_dist(next_x_dists, next_x,
                                                                  keep_variable_dim=True,
                                                                  loss_type="recon")
                    # # (bs, seq_len - 3, num_hiddens)
                    # masked_recon_h = self.prediction_loss_from_dist(next_z_prior_dists, next_z_infer_feature,
                    #                                                 keep_variable_dim=True,
                    #                                                 loss_type="recon_h")
                next_z_prior_masked_.append(next_z_prior_masked)
                masked_recon_.append(masked_recon)
                # masked_recon_h_.append(masked_recon_h)
            eval_recon = eval_recon.sum(dim=-1).mean()  # scalar
            eval_details["eval_recon"] = eval_recon
            rew_loss, rew_loss_detail = self.rew_loss_from_feature(st, r)

        # clean cache
        self.use_cache = False
        self.sa_feature_cache = None
        self.action_feature = None
        self.full_state_feature = None

        # log(T_o(full) / T_o(masked))
        # (bs, seq_len - 2, num_observables, 1)
        full_recon = full_recon.unsqueeze(-1)
        # (bs, seq_len - 2, num_observables, feature_dim + 1)
        masked_recon_ = torch.stack(masked_recon_, dim=-1)
        CMI_o = masked_recon_ - full_recon
        # (num_observables, feature_dim + 1)
        CMI_o_mean = CMI_o.mean(dim=(0, 1))
        # standard error of the mean
        CMI_o_sem = CMI_o.std(dim=(0, 1)) / np.sqrt(np.prod(CMI_o.shape[:2]))

        # # log(T_h(full) / T_h(masked))
        # # (bs, seq_len - 3, num_hiddens, 1)
        # full_recon_h = full_recon_h.unsqueeze(-1)
        # # (bs, seq_len - 3, num_hiddens, feature_dim + 1)
        # masked_recon_h_ = torch.stack(masked_recon_h_, dim=-1)
        # CMI_h = masked_recon_h_ - full_recon_h
        # # (num_hiddens, feature_dim + 1)
        # CMI_h_mean = CMI_h.mean(dim=(0, 1))
        # CMI_h_sem = CMI_h.std(dim=(0, 1)) / np.sqrt(np.prod(CMI_h.shape[:2]))

        # KL(T_h(full) || T_h(masked))
        # (bs, seq_len - 2, num_hidden, 1, num_colors)
        next_z_prior_full = next_z_prior_full.unsqueeze(-2)
        # (bs, seq_len - 2, num_hidden, feature_dim + 1, num_colors)
        next_z_prior_masked_ = torch.stack(next_z_prior_masked_, dim=-2)
        # summed over num_colors for KL as CMI of the hidden
        # (bs, seq_len - 2, num_hidden, feature_dim + 1)
        CMI_h = torch.sum(next_z_prior_full * (torch.log(next_z_prior_full + 1e-20) -
                                               torch.log(next_z_prior_masked_ + 1e-20)), dim=-1)
        # (num_hidden, feature_dim + 1)
        CMI_h_mean = CMI_h.mean(dim=(0, 1))

        # (feature_dim, feature_dim + 1)
        CMI_mean = torch.cat((CMI_o_mean[:self.hidden_objects_ind[0]],
                              CMI_h_mean, CMI_o_mean[self.hidden_objects_ind[0]:]), dim=0)
        # CMI_sem = torch.cat((CMI_o_sem[:self.hidden_objects_ind[0]],
        #                      CMI_h_sem, CMI_o_sem[self.hidden_objects_ind[0]:]), dim=0)
        # CMI_lb = CMI_mean - 2 * CMI_sem
        # CMI_ub = CMI_mean + 2 * CMI_sem
        # print("CMI_mean", CMI_mean)
        # print("CMI_sem", CMI_sem)

        # self.eval_step_CMI_lb += CMI_lb
        # self.eval_step_CMI_ub += CMI_ub
        self.eval_step_CMI += CMI_mean
        self.mask_update_idx += 1

        eval_steps = self.cmi_params.eval_steps
        eval_tau = self.cmi_params.eval_tau
        if self.mask_update_idx == eval_steps:
            # self.eval_step_CMI_lb /= eval_steps
            # self.eval_step_CMI_ub /= eval_steps
            # self.mask_CMI_lb = self.mask_CMI_lb * eval_tau + self.eval_step_CMI_lb * (1 - eval_tau)
            # self.mask_CMI_ub = self.mask_CMI_ub * eval_tau + self.eval_step_CMI_ub * (1 - eval_tau)
            #
            # mask_linked = self.mask_CMI_lb > self.CMI_threshold
            # mask_unlinked = self.mask_CMI_ub < self.CMI_threshold
            # self.mask = ~mask_unlinked
            # print("mask_linked", mask_linked)
            # print("mask_unlinked", mask_unlinked)

            self.eval_step_CMI /= eval_steps
            self.mask_CMI = self.mask_CMI * eval_tau + self.eval_step_CMI * (1 - eval_tau)

            # ensure each object has at least one parent
            mask_high_thres = self.mask_CMI >= self.CMI_threshold
            mask_CMI_max, _ = self.mask_CMI.max(dim=1, keepdim=True)
            mask_low_thres = (self.mask_CMI < self.CMI_threshold) * (self.mask_CMI == mask_CMI_max)
            self.mask = mask_high_thres + mask_low_thres

        if not self.update_num_eval % self.print_eval_freq:
            print("z_infer_probs", z_infer_probs[:5])
            print("next_z_prior_full", next_z_prior_full[:5])

            print("next_true_hidden", next_hidden[:20].t())
            print("next_enco_hidden", next_feature_hidden[:20].t())
            print("next_pred_hidden", pred_next_feature_hidden[:20].t())

            # print("mask_CMI_lb", self.mask_CMI_lb)
            # print("mask_CMI_ub", self.mask_CMI_ub)
            print("mask_CMI", self.mask_CMI)

        loss_details = {**eval_details, **rew_loss_detail}
        return loss_details

    def reward(self, obs, actions, next_obses, output_numpy=False):
        """
        Calculate reward for RL policy
        :param obs: (bs, obs_spec) during policy training or (obs_spec,) after env.step()
        :param actions: (bs, n_pred_step, action_dim) during policy training or (action_dim,) after env.step()
        :param next_obses: (bs, n_pred_step, obs_spec) during policy training or (obs_spec,) after env.step()
        :param output_numpy: output numpy or tensor
        :return: (bs, n_pred_step, 1) or scalar
        """
        obs, actions, next_obses, reward_need_squeeze = self.preprocess(obs, actions, next_obses)

        with torch.no_grad():
            full_next_dist, causal_next_dist = self.forward(obs, actions, forward_mode=("full", "causal",))
            next_features = self.encoder(next_obses)
            full_neg_log_prob = self.prediction_loss_from_dist(full_next_dist, next_features)  # (bs, n_pred_step)
            causal_neg_log_prob = self.prediction_loss_from_dist(causal_next_dist, next_features)  # (bs, n_pred_step)

            causal_pred_reward = full_neg_log_prob

            normalized_causal_pred_reward = torch.tanh((causal_pred_reward - self.causal_pred_reward_mean) /
                                                       (self.causal_pred_reward_std * 2))

            tau = 0.99
            if len(causal_pred_reward) > 0:
                batch_mean = causal_pred_reward.mean(dim=0)
                batch_std = causal_pred_reward.std(dim=0, unbiased=False)
                self.causal_pred_reward_mean = self.causal_pred_reward_mean * tau + batch_mean * (1 - tau)
                self.causal_pred_reward_std = self.causal_pred_reward_std * tau + batch_std * (1 - tau)

            pred_diff_reward = causal_neg_log_prob - full_neg_log_prob  # (bs, n_pred_step)
            normalized_pred_diff_reward = torch.tanh(pred_diff_reward / (self.pred_diff_reward_std * 2))
            if len(pred_diff_reward) > 0:
                batch_std = pred_diff_reward.std(dim=0, unbiased=False)
                self.pred_diff_reward_std = self.pred_diff_reward_std * tau + batch_std * (1 - tau)

            causal_pred_reward_weight = self.cmi_params.causal_pred_reward_weight
            pred_diff_reward_weight = self.cmi_params.pred_diff_reward_weight
            reward = causal_pred_reward_weight * normalized_causal_pred_reward + \
                     pred_diff_reward_weight * normalized_pred_diff_reward

            reward = reward[..., None]  # (bs, n_pred_step, 1)

        reward = self.reward_postprocess(reward, reward_need_squeeze, output_numpy)

        return reward

    def eval_prediction(self, obs, actions, next_obses):
        obs, actions, next_obses, _ = self.preprocess(obs, actions, next_obses)

        with torch.no_grad():
            feature = self.encoder(obs)
            next_feature = self.encoder(next_obses)
            pred_next_dist = self.forward_with_feature(feature, actions, forward_mode=("causal",))
            pred_loss = self.prediction_loss_from_dist(pred_next_dist, next_feature)

            accuracy = None
            if not self.continuous_state:
                accuracy = []
                for dist_i, next_feature_i in zip(pred_next_dist, next_feature):
                    if not isinstance(dist_i, OneHotCategorical):
                        continue
                    logits = dist_i.logits  # (bs, n_pred_step, feature_i_inner_dim)
                    # (bs, n_pred_step)
                    accuracy_i = logits.argmax(dim=-1) == next_feature_i.argmax(dim=-1)
                    accuracy.append(accuracy_i)
                accuracy = torch.stack(accuracy, dim=-1)
                accuracy = to_numpy(accuracy)

        return pred_next_dist, next_feature, pred_loss, accuracy

    def get_mask(self):
        return self.mask

    def get_state_abstraction(self):
        self.abstraction_quested = True
        abstraction_graph = self.update_abstraction()
        self.update_abstracted_dynamics()
        return abstraction_graph

    def update_abstraction(self):
        self.abstraction_graph = get_state_abstraction(to_numpy(self.get_mask()))
        self.abstraction_idxes = list(self.abstraction_graph.keys())

        action_idx = self.feature_dim
        self.action_children_idxes = [idx for idx, parent_idxes in self.abstraction_graph.items()
                                      if action_idx in parent_idxes]
        self.abstraction_adjacency = {}
        for idx, parents in self.abstraction_graph.items():
            self.abstraction_adjacency[idx] = [parent for parent in parents if parent < action_idx]

        return self.abstraction_graph

    def update_abstracted_dynamics(self, ):
        # only need to calculate action feature for state variables that are children of the action
        action_children_idxes = self.action_children_idxes
        self.abstraction_action_feature_weights = [w[action_children_idxes]
                                                   for w in self.action_feature_weights]
        self.abstraction_action_feature_biases = [b[action_children_idxes]
                                                  for b in self.action_feature_biases]

        # when predicting each state variables in the abstraction, only need to compute state feature for their parents
        feature_dim = self.feature_dim
        self.abstraction_state_feature_1st_layer_weights = {}
        self.abstraction_state_feature_1st_layer_biases = {}
        idxes = []
        for idx, parent_idxes in self.abstraction_adjacency.items():
            idxes.extend([parent_idx + idx * feature_dim for parent_idx in parent_idxes])
            self.abstraction_state_feature_1st_layer_weights[idx] = \
                [w[idx:idx + 1] for i, w in enumerate(self.state_feature_1st_layer_weights) if i in parent_idxes]
            self.abstraction_state_feature_1st_layer_biases[idx] = \
                [b[idx:idx + 1] for i, b in enumerate(self.state_feature_1st_layer_biases) if i in parent_idxes]

        self.abstraction_state_feature_weights = [w[idxes] for w in self.state_feature_weights]
        self.abstraction_state_feature_biases = [b[idxes] for b in self.state_feature_biases]

        abstraction_idxes = self.abstraction_idxes
        self.abstraction_generative_weights = [w[abstraction_idxes] for w in self.generative_weights]
        self.abstraction_generative_biases = [b[abstraction_idxes] for b in self.generative_biases]
        self.abstraction_generative_last_layer_weights = \
            [w for i, w in enumerate(self.generative_last_layer_weights) if i in abstraction_idxes]
        self.abstraction_generative_last_layer_biases = \
            [b for i, b in enumerate(self.generative_last_layer_biases) if i in abstraction_idxes]

    def get_adjacency(self):
        # return self.mask_CMI_lb[:, :-1], self.mask_CMI_ub[:, :-1]
        return self.mask_CMI[:, :-1]

    def get_intervention_mask(self):
        # return self.mask_CMI_lb[:, -1:], self.mask_CMI_ub[:, -1:]
        return self.mask_CMI[:, -1:]

    def train(self, training=True):
        self.training = training
        # self.encoder.eval()  # Freeze encoder's parameters

    def eval(self):
        self.train(training=False)

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "mask_CMI": self.mask_CMI,
                    # "mask_CMI_lb": self.mask_CMI_lb,
                    # "mask_CMI_ub": self.mask_CMI_ub,
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("inference loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.mask_CMI = checkpoint["mask_CMI"]
            self.mask = self.mask_CMI >= self.CMI_threshold
            # self.mask_CMI_lb = checkpoint["mask_CMI_lb"]
            # self.mask_CMI_ub = checkpoint["mask_CMI_ub"]
            # self.mask = ~(self.mask_CMI_ub < self.CMI_threshold)
            # self.mask_CMI[self.diag_mask] = self.CMI_threshold
            # self.mask[self.diag_mask] = True
