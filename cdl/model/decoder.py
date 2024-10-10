import sys
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.inference_utils import reset_layer


class FeedforwardDecoder(nn.Module):
    def __init__(self, params, feedforward_dec_params):
        super().__init__()
        self.params = params
        self.device = params.device

        chemical_env_params = params.env_params.chemical_env_params
        self.reward_type = chemical_env_params.reward_type
        self.num_objects = chemical_env_params.num_objects
        self.num_colors = chemical_env_params.num_colors
        hidden_objects_ind = chemical_env_params.hidden_objects_ind
        self.hidden_target_ind = hidden_objects_ind + [i + self.num_objects for i in hidden_objects_ind]
        self.ff_dec_params = feedforward_dec_params
        self.categorical_rew = feedforward_dec_params.categorical_rew

        self.init_model()
        self.to(self.device)

    def init_model(self):
        dim_st = 2 * self.num_objects * self.num_colors
        # dim_st = len(self.hidden_target_ind) * self.num_colors
        if self.categorical_rew:
            dim_rew = (self.num_objects + 1 if self.reward_type == "match"
                       else (self.num_colors - 1) * self.num_objects + 1)
            # dim_rew = self.num_objects + 1 if self.reward_type == "match" else self.num_colors
        else:
            dim_rew = 1
        dims_h = self.ff_dec_params.dims_h
        dropout_p = self.ff_dec_params.dropout_p
        dic_layers = OrderedDict()
        for n in range(len(dims_h)):
            if n == 0:
                dic_layers['linear' + str(n)] = nn.Linear(dim_st, dims_h[n])
            else:
                dic_layers['linear' + str(n)] = nn.Linear(dims_h[n - 1], dims_h[n])
            dic_layers['layer_norm' + str(n)] = nn.LayerNorm(dims_h[n])
            dic_layers['activation' + str(n)] = nn.ReLU()
            dic_layers['dropout' + str(n)] = nn.Dropout(p=dropout_p)
        dic_layers['linear_last'] = nn.Linear(dims_h[-1], dim_rew)
        self.mlp = nn.Sequential(dic_layers)

        # apply the same initialization scheme to linear layers in the decoder
        # as those in the transition model
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                reset_layer(layer.weight, layer.bias)

    def forward(self, feature):
        """
        :param feature: [(bs, n_pred_step, num_colors)] * (2 * num_objects)
        :return rew: (bs, n_pred_step, dim_rew)
        """
        # # [(bs, n_pred_step, num_colors)] * (2 * num_hiddens)
        # feature = [feature[i] for i in self.hidden_target_ind]
        # # (bs, n_pred_step, 2 * num_hiddens * num_colors)
        # feature = torch.cat(feature, dim=-1)

        # (bs, n_pred_step, 2 * num_objects * num_colors)
        feature = torch.cat(feature, dim=-1)

        # (bs, n_pred_step, dim_rew)
        rew = self.mlp(feature)
        return rew


def rew_decoder(params):
    decoder_type = params.decoder_params.decoder_type
    if decoder_type == "feedforward":
        feedforward_dec_params = params.decoder_params.feedforward_dec_params
        decoder = FeedforwardDecoder(params, feedforward_dec_params)
    else:
        raise NotImplementedError("Unknown decoder_type: {}".format(decoder_type))
    return decoder
