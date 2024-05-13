import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.inference_utils import reset_layer


class FeedforwardDecoder(nn.Module):
    def __init__(self, params, input_dim, output_dim, hidden_dim, dropout, categorical_rew):
        super().__init__()
        self.categorical_rew = categorical_rew
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # apply the same initialization scheme to linear layers in the decoder
        # as those in the transition model
        for layer in self.feedforward:
            if isinstance(layer, nn.Linear):
                reset_layer(layer.weight, layer.bias)

    def forward(self, feature):
        """
        :param feature: [(bs, n_pred_step, num_colors)] * (2 * num_objects)
        :return rew: (bs, n_pred_step, reward_dim)
        """
        feature = torch.stack(feature, dim=-2)  # (bs, n_pred_step, 2 * num_objects, num_colors)
        feature = [feature_i.reshape((feature.shape[0], -1))
                   for feature_i in
                   torch.unbind(feature, dim=1)]  # [(bs, 2 * num_objects * num_colors)] * n_pred_step
        rew = []
        for feature_i in feature:
            rew_i = self.feedforward(feature_i)  # (bs, reward_dim)
            rew.append(rew_i)  # [(bs, reward_dim)] * n_pred_step
        rew = torch.stack(rew, dim=-2)  # (bs, n_pred_step, reward_dim)
        return rew


def rew_decoder(params):
    decoder_type = params.decoder_params.decoder_type
    if decoder_type == "feedforward":
        feedforward_dec_params = params.decoder_params.feedforward_dec_params
        chemical_env_params = params.env_params.chemical_env_params
        obs_shape = 2 * chemical_env_params.num_objects * chemical_env_params.num_colors
        logit_shape = (chemical_env_params.num_objects + 1) if feedforward_dec_params.categorical_rew else 1
        device = params.device
        hidden_layer_size = feedforward_dec_params.hidden_layer_size
        dropout = feedforward_dec_params.dropout
        categorical_rew = feedforward_dec_params.categorical_rew
        decoder = FeedforwardDecoder(params, obs_shape, logit_shape, hidden_layer_size, dropout, categorical_rew).to(device)
    else:
        raise ValueError("Unknown decoder_type: {}".format(decoder_type))
    return decoder
