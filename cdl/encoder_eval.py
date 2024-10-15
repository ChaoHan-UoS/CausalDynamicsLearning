import os
import sys

from collections import OrderedDict
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=5, sci_mode=False)

from model.random_policy import RandomPolicy
from model.inference_cmi import InferenceCMI
from model.encoder import Encoder
from model.decoder import rew_decoder

from utils.utils import (TrainingParams, update_obs_act_spec, set_seed_everywhere, get_env,
                         get_start_step_from_model_loading, preprocess_obs, postprocess_obs)
from utils.scripted_policy import get_scripted_policy, get_is_demo

from tianshou.data import Batch, ReplayBuffer, to_torch

from env.chemical_env import Chemical

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')


def batch_process(batch_data, params):
    """
    :return obs_batch: Batch(obs_i_key: (bs, seq_len, obs_i_shape))
            hidden_batch: {obs_i_key: (bs, seq_len, obs_i_shape)}
    """
    batch_data = to_torch(batch_data, torch.float64, params.device)
    obs_batch = batch_data.obs
    hidden_batch = {key: batch_data.info[key] for key in params.hidden_keys}
    return obs_batch, hidden_batch

# supervised hidden decoder
class HiddenDecoder(nn.Module):
    def __init__(self, encoder, params):
        super().__init__()
        self.encoder = encoder
        self.params = params
        self.device = params.device
        self.decoder_params = params.hidden_decoder_params
        self.num_colors = params.env_params.chemical_env_params.num_colors
        self.num_hiddens = len(params.hidden_keys)

        self.init_model()
        self.to(self.device)

        self.loss_ce = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.decoder_params.lr)

    def init_model(self):
        linear = self.decoder_params.linear
        dims_h = self.decoder_params.dims_h
        dropout_p = self.decoder_params.dropout_p
        self.mlp = nn.ModuleList()
        for i in range(self.num_hiddens):
            dic_layers = OrderedDict()
            if linear:
                dic_layers['linear'] = nn.Linear(self.num_colors * self.num_hiddens, self.num_colors)
            else:
                for n in range(len(dims_h)):
                    if n == 0:
                        dic_layers['linear' + str(n)] = nn.Linear(self.num_colors * self.num_hiddens, dims_h[n])
                    else:
                        dic_layers['linear' + str(n)] = nn.Linear(dims_h[n - 1], dims_h[n])
                    dic_layers['layer_norm' + str(n)] = nn.LayerNorm(dims_h[n])
                    dic_layers['activation' + str(n)] = nn.ReLU()
                    dic_layers['dropout' + str(n)] = nn.Dropout(p=dropout_p)
                dic_layers['linear_last'] = nn.Linear(dims_h[-1], self.num_colors)
            self.mlp.append(nn.Sequential(dic_layers))

    def forward(self, input):
        """
        :param input: (bs, seq_len - 1, num_colors * num_hiddens)
        :return output: (bs, num_colors, seq_len - 1, num_hiddens)
        """
        output = []
        for i in range(self.num_hiddens):
            output.append(self.mlp[i](input))
        output = torch.stack(output, dim=-1).transpose(1, 2)
        return output

    def backprop(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, obs_batch, hidden_batch):
        """
        :param obs_batch: Batch(obs_i_key: (bs, seq_len, obs_i_shape))
        :param hidden_batch: {obs_i_key: (bs, seq_len, obs_i_shape)}
        :return recon_loss: scalar
                recon_loss_detail: dict
        """
        # (bs, seq_len, num_hiddens * num_colors)
        hidden_enc = self.encoder(obs_batch)[0]
        # (bs, num_colors, seq_len - 1, num_hiddens)
        hidden_dec = self.forward(hidden_enc[:, 2:])

        # hidden from t=1 to T-1
        hidden_label = [hidden_batch[key][:, 1:-1].squeeze(dim=-1).long()
                        for key in self.params.hidden_keys]
        # (bs, seq_len - 1, num_hiddens)
        hidden_label = torch.stack(hidden_label, dim=-1)

        if self.training:
            # (bs, seq_len - 1, num_hiddens)
            recon_loss = self.loss_ce(hidden_dec, hidden_label)
            recon_loss = recon_loss.sum((1, 2)).mean()
            self.backprop(recon_loss)
        else:
            recon_loss = (hidden_dec.max(dim=1)[1] == hidden_label).float().mean()

        recon_loss_detail = {"recon_loss": recon_loss}
        return recon_loss, recon_loss_detail

def pred_state_acc(next_obs, next_hidden, pred_obs_dists, pred_hidden_dists, keep_variable_dim=False):
    """
    :param next_obs/next_hidden: [(bs, seq_len - 2, num_colors)] * num_observables/num_hiddens
    :param pred_obs_dists: [OneHotCategorical(bs, seq_len - 1, num_colors)] * num_observables
    :param pred_hidden_dists: [(bs, seq_len - 1, num_colors)] * num_hiddens
    :param keep_variable_dim: whether to keep the dimension of state variables which is dim=-1

    """
    # states from t=2 to T-1
    # (bs, seq_len - 2, num_observables/num_hiddens)
    next_obs = torch.stack(next_obs, -1).argmax(-2)
    next_hidden = torch.stack(next_hidden, -1).argmax(-2)
    pred_obs = torch.stack([dist_i.probs[:, :].argmax(-1) for dist_i in pred_obs_dists], -1)
    pred_hidden = torch.stack(pred_hidden_dists, -1)[:, :].argmax(-2)

    mean_dim = (0, 1) if keep_variable_dim else None
    pred_obs_acc = (pred_obs == next_obs).float().mean(dim=mean_dim)
    pred_hidden_acc = (pred_hidden == next_hidden).float().mean(dim=mean_dim)
    return pred_obs_acc, pred_hidden_acc

def train(params):
    device = torch.device("cuda:{}".format(params.cuda_id) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    set_seed_everywhere(params.seed)

    params.device = device
    training_params = params.training_params
    inference_params = params.inference_params
    cmi_params = inference_params.cmi_params
    encoder_params = params.encoder_params
    gumbel_softmax_params = encoder_params.gumbel_softmax_params

    # init environment
    chemical_env_params = params.env_params.chemical_env_params
    num_objects = chemical_env_params.num_objects
    hidden_objects_ind = chemical_env_params.hidden_objects_ind
    hidden_targets_ind = chemical_env_params.hidden_targets_ind
    ind_f = range(2 * num_objects)
    params.hidden_ind = hidden_objects_ind + [num_objects + i for i in hidden_targets_ind]
    params.obs_ind = [i for i in ind_f if i not in params.hidden_ind]
    assert 0 < len(params.obs_ind) <= len(ind_f)
    params.keys_f = params.obs_keys_f + params.goal_keys_f
    params.obs_keys = [params.keys_f[i] for i in params.obs_ind]
    params.hidden_keys = [k for k in params.keys_f if k not in params.obs_keys]

    render = False
    num_env = params.env_params.num_env
    is_vecenv = num_env > 1
    env = get_env(params, render)

    # init model
    update_obs_act_spec(env, params)
    encoder = Encoder(params)
    decoder = rew_decoder(params)
    hdecoder = HiddenDecoder(encoder, params)

    inference_algo = params.training_params.inference_algo
    use_cmi = inference_algo == "cmi"
    if inference_algo == "cmi":
        Inference = InferenceCMI
    else:
        raise NotImplementedError
    inference = Inference(encoder, decoder, params)
    inference.eval()
    encoder.eval()
    decoder.eval()
    hdecoder.encoder.eval()
    print(f'mask_CMI: {inference.mask_CMI} \n')

    scripted_policy = get_scripted_policy(env, params)
    rl_algo = params.training_params.rl_algo
    is_task_learning = rl_algo == "model_based"
    if rl_algo == "random":
        policy = RandomPolicy(params)
    else:
        raise NotImplementedError

    # init replay buffer
    replay_buffer_params = training_params.replay_buffer_params
    buffer_train = ReplayBuffer(
        size=replay_buffer_params.capacity,
        stack_num=replay_buffer_params.stack_num,
        ignore_obs_next=True,
        sample_avail=True,
    )
    buffer_eval = ReplayBuffer(
        size=replay_buffer_params.capacity,
        stack_num=replay_buffer_params.stack_num,
        ignore_obs_next=True,
        sample_avail=True,
    )

    # init saving
    writer = SummaryWriter(os.path.join(params.rslts_dir, "tensorboard"))
    plot_dir = os.path.join(params.rslts_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    start_step = 0
    total_steps = 30000
    collect_env_step = training_params.collect_env_step
    supervised_hidden_decoder = training_params.supervised_hidden_decoder
    train_prop = inference_params.train_prop

    # init episode variables
    episode_num = 0
    (obs, obs_f), info = env.reset()
    scripted_policy.reset(obs)

    done = np.zeros(num_env, dtype=bool) if is_vecenv else False
    success = False
    episode_reward = np.zeros(num_env) if is_vecenv else 0
    episode_step = np.zeros(num_env) if is_vecenv else 0
    is_train = (np.random.rand(num_env) if is_vecenv else np.random.rand()) < train_prop
    is_demo = np.array([get_is_demo(0, params) for _ in range(num_env)]) if is_vecenv else get_is_demo(0, params)

    for step in range(start_step, total_steps):
        is_init_stage = step < training_params.init_steps
        if (step + 1) % 100 == 0:
            print("{}/{}, init_stage: {}".format(step + 1, total_steps, is_init_stage))
        loss_details = {"hidden_decoder_train": []}

        # env interaction and transition saving
        if collect_env_step:
            # reset in the beginning of an episode
            if is_vecenv and done.any():
                for i, done_ in enumerate(done):
                    if not done_:
                        continue
                    is_train[i] = np.random.rand() < train_prop
                    is_demo[i] = get_is_demo(step, params)
                    scripted_policy.reset(obs, i)

                    episode_reward[i] = 0
                    episode_step[i] = 0
                    episode_num += 1
            elif not is_vecenv and done:
                (obs, obs_f), _ = env.reset()
                scripted_policy.reset(obs)

                is_train = np.random.rand() < train_prop
                is_demo = get_is_demo(step, params)
                episode_reward = 0
                episode_step = 0
                success = False
                episode_num += 1

            # get action
            inference.eval()
            policy.eval()
            if is_init_stage:
                if is_vecenv:
                    action = np.array([policy.act_randomly() for _ in range(num_env)])
                else:
                    action = policy.act_randomly()
            else:
                if is_vecenv:
                    action = policy.act(obs)
                    if is_demo.any():
                        demo_action = scripted_policy.act(obs)
                        action[is_demo] = demo_action[is_demo]
                else:
                    action_policy = scripted_policy if is_demo else policy
                    action = action_policy.act(obs)

            (next_obs, next_obs_f), env_reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated

            if is_task_learning and not is_vecenv:
                success = success or info["success"]

            inference_reward = np.zeros(num_env) if is_vecenv else 0
            episode_reward += env_reward if is_task_learning else inference_reward
            episode_step += 1

            hidden_state = {key: obs_f[key] for key in params.hidden_keys}
            info.update(hidden_state)
            obs['act'], obs['rew'] = np.array([action]), np.array([env_reward])  # [s, a, r] -> s

            # is_train: if the transition is training data or evaluation data for supervised hidden decoder
            if is_train:
                buffer_train.add(
                    Batch(obs=obs,
                          act=action,
                          rew=env_reward,
                          terminated=terminated,
                          truncated=truncated,
                          info=info,
                          )
                )
            else:
                buffer_eval.add(
                    Batch(obs=obs,
                          act=action,
                          rew=env_reward,
                          terminated=terminated,
                          truncated=truncated,
                          info=info,
                          )
                )

            obs = next_obs
            obs_f = next_obs_f

        # training
        if is_init_stage:
            continue

        if supervised_hidden_decoder:
            batch_data, batch_ids = buffer_train.sample(inference_params.batch_size)
            obs_batch, hidden_batch = batch_process(batch_data, params)

            # obs -> encoded hidden -> decoded hidden <-> hidden label
            recon_loss, recon_loss_detail = hdecoder.update(obs_batch, hidden_batch)
            loss_details["hidden_decoder_train"].append(recon_loss_detail)
            if (step + 1) % 500 == 0:
                print("{}/{}, recon_loss: {:.4f}".format(step + 1, total_steps, recon_loss.item()))

        # logging
        if writer is not None:
            for module_name, module_loss_detail in loss_details.items():
                if not module_loss_detail:
                    continue
                # list of dict to dict of list
                if isinstance(module_loss_detail, list):
                    keys = set().union(*[dic.keys() for dic in module_loss_detail])
                    module_loss_detail = {k: [dic[k] for dic in module_loss_detail if k in dic]
                                          for k in keys if k not in ["priority"]}
                for loss_name, loss_values in module_loss_detail.items():
                    writer.add_scalar("{}/{}".format(module_name, loss_name),
                                      torch.mean(torch.stack(loss_values)), step)

    with torch.no_grad():
        # evaluate edge accuracy
        intervention_mask = torch.ones(num_objects)
        intervention_mask[hidden_objects_ind] = 0
        adjacency_intervention = torch.cat((env.adjacency_matrix, intervention_mask.unsqueeze(-1)),
                                           dim=-1).to(device)
        edge_acc = (adjacency_intervention == inference.mask).float().mean()
        print(f'edge_acc: {edge_acc} \n')

        batch_data, batch_ids = buffer_eval.sample(inference_params.batch_size)
        obs_batch, hidden_batch = batch_process(batch_data, params)

        # evaluate hidden decoding accuracy
        if supervised_hidden_decoder:
            hdecoder.eval()
            recon_acc, recon_acc_detail = hdecoder.update(obs_batch, hidden_batch)
            print(f'hidden decoding accuracy: {recon_acc} \n')

        # evaluate predicted observables and hiddens accuracy
        # (bs, seq_len, num_hiddens/num_observables * num_colors)
        hidden_enc, _, obs, act, st, r = encoder(obs_batch)
        # states from t=2 to T-1
        # [(bs, seq_len - 2, num_colors)] * num_observables/num_hiddens
        next_obs, next_hidden = inference.get_next_state(obs, hidden_enc)
        # predicted states from t=2 to T
        pred_obs_dists, pred_hidden_dists = inference.forward_with_feature(obs, hidden_enc, act,
                                                                           forward_mode=("causal",))
        pred_obs_acc, pred_hidden_acc = pred_state_acc(next_obs, next_hidden,
                                                       pred_obs_dists, pred_hidden_dists, keep_variable_dim=True)
        print(f'predicted observables accuracy: {pred_obs_acc}')
        print(f'predicted hiddens accuracy: {pred_hidden_acc} \n')

        # evaluate reward prediction accuracy
        rew_loss, rew_loss_detail = inference.rew_loss_from_feature(st, r)
        print(f'reward prediction accuracy: {rew_loss}')

if __name__ == "__main__":
    rslts_dir = "/home/chao/PycharmProjects/CausalDynamicsLearning-DVAE/rslts/dynamics/noisy_o1_allpast_history_encoder_2024_10_15_22_04_13"
    params_path = os.path.join(rslts_dir, "params.json")
    params = TrainingParams(training_params_fname=params_path, train=False)
    params.rslts_dir = rslts_dir
    train(params)
