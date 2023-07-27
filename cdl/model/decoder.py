import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# class LinearLayer:
#     def __init__(self, input_size, output_size):
#         self.weights = torch.randn(output_size, input_size)
#         self.bias = torch.randn(output_size)
#
#     def forward(self, x):
#         return torch.matmul(x, self.weights.t()) + self.bias


# # Randomly initialize weights and bias
# weights = np.random.randn(128, 18)
# bias = np.random.randn(128)
#
# # Create the Linear layer
# class FeedforwardDecoder:
#     def __init__(self, params, input_dim, output_dim, hidden_dim):
#         self.weights = weights
#         self.bias = bias
#         self.input_size = input_dim
#         self.output_size = output_dim
#
#     def __call__(self, x):
#         return np.dot(x, self.weights.T) + self.bias


class FeedforwardDecoder(nn.Module):
    def __init__(self, params, input_dim, output_dim, hidden_dim):
        super().__init__()
        # self.new = nn.LSTM(18, 128)
        # self.feedforward1 = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim)
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, output_dim)
        # )

        print('REWARD INIT')

    def forward(self, feature):
        """
        :param feature: [(bs, (n_pred_step), num_colors)] * (2 * num_objects)
        :return: (bs, (n_pred_step), reward_dim)
        """
        """
        feature = torch.stack(feature, dim=0)  # (2 * num_objects, bs, (n_pred_step), num_colors)

        feature_dims = len(feature.shape)
        if feature_dims == 4:
            feature = feature.permute(1, 0, 2, 3)  # (bs, 2 * num_objects, n_pred_step, num_colors)
            feature_pred_step = [feature_i.reshape((feature.shape[0], -1))
                                 for feature_i in
                                 torch.unbind(feature, dim=-2)]  # [(bs, 2 * num_objects * num_colors)] * n_pred_step
            feature_dec = []
            for feature_i in feature_pred_step:
                feature_dec_i = self.feedforward(feature_i) # (bs, reward_dim)
                feature_dec.append(feature_dec_i)  # [(bs, reward_dim)] * n_pred_step
            feature_dec = torch.stack(feature_dec, dim=-2)  # (bs, n_pred_step, reward_dim)
        else:
            feature = feature.permute(1, 0, 2)  # (bs, 2 * num_objects, num_colors)
            feature = feature.reshape((feature.shape[0], -1))  # (bs, 2 * num_objects * num_colors)
            feature_dec = self.feedforward(feature)  # (bs, reward_dim)
        """

        return None



def rew_decoder(params):
    decoder_type = params.decoder_params.decoder_type
    if decoder_type == "feedforward":
        feedforward_dec_params = params.decoder_params.feedforward_dec_params
        chemical_env_params = params.env_params.chemical_env_params
        obs_shape = 2 * chemical_env_params.num_objects * chemical_env_params.num_colors
        logit_shape = 1
        device = params.device
        hidden_layer_size = feedforward_dec_params.hidden_layer_size
        decoder1 = FeedforwardDecoder(params, obs_shape, logit_shape, hidden_layer_size)

        # # Example usage
        # input_size = 18
        # output_size = 128
        #
        # # Create the Linear layer
        # fc1 = LinearLayer(input_size, output_size)

    else:
        raise ValueError("Unknown decoder_type: {}".format(decoder_type))

    print('REWARD DECODER')
    return None




