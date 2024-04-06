import os
import sys

os.environ['JUPYTER_PLATFORM_DIRS'] = '1'

from collections import Counter
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=5, sci_mode=False)

from model.inference_mlp import InferenceMLP
from model.inference_gnn import InferenceGNN
from model.inference_reg import InferenceReg
from model.inference_nps import InferenceNPS
from model.inference_cmi import InferenceCMI
from model.inference_utils import obs_batch_dict2tuple, count2prob, count2prob_llh

from model.random_policy import RandomPolicy
from model.hippo import HiPPO
from model.model_based import ModelBased

from model.encoder import obs_encoder
from model.decoder import rew_decoder

from utils.utils import TrainingParams, update_obs_act_spec, set_seed_everywhere, get_env, \
    get_start_step_from_model_loading, preprocess_obs, postprocess_obs
# from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # use tianshou's buffer instead
from utils.plot import plot_adjacency_intervention_mask
from utils.scripted_policy import get_scripted_policy, get_is_demo

from tianshou.data import Batch, ReplayBuffer, PrioritizedReplayBuffer
from tianshou.data import to_torch

from env.chemical_env import Chemical

import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('TkAgg')


def sample_process(batch_data, params):
    """
    :return obs_batch: Batch(obs_i_key: (bs, stack_num, obs_i_shape))
            actions_batch: (bs, n_pred_step, action_dim)
            next_obses_batch: Batch(obs_i_key: (bs, stack_num, n_pred_step, obs_i_shape))
            rew_batch: (bs, reward_dim) or None
            next_rews_batch: (bs, n_pred_step, reward_dim)
            hidden_label_batch: (bs, hidden_dim)
    """
    replay_buffer_params = params.training_params.replay_buffer_params
    inference_params = params.inference_params

    batch_data = to_torch(batch_data, torch.float32, params.device)
    # Batch(obs_i_key: (bs, stack_num, obs_i_shape))
    obs_batch = batch_data.obs[:, :replay_buffer_params.stack_num]
    # (bs, reward_dim)
    # observation at t corresponds to reward at t-1
    if replay_buffer_params.stack_num == 1:
        rew_batch = None
    else:
        rew_batch = obs_batch.rew[:, -3]

    # {obs_i_key: (bs, obs_i_shape)}
    next_hidden_batch = {key: batch_data.info[key][:, replay_buffer_params.stack_num - 1]
                         for key in params.hidden_keys}

    actions_batch = []
    next_obses_batch = []
    next_rews_batch = []
    for i in range(inference_params.n_pred_step):
        actions_batch.append(batch_data.obs.act[:, i + replay_buffer_params.stack_num - 2])
        next_obses_batch.append(batch_data.obs_next[:, i: i + replay_buffer_params.stack_num])
        if replay_buffer_params.stack_num == 1 and i == 0:
            next_rews_batch.append(batch_data.obs.rew[:, replay_buffer_params.stack_num - 1])
        else:
            next_rews_batch.append(batch_data.obs_next.rew[:, i + replay_buffer_params.stack_num - 3])
    # (bs, n_pred_step, action_dim)
    actions_batch = torch.stack(actions_batch, dim=-2)
    # Batch(obs_i_key: (bs, stack_num, n_pred_step, obs_i_shape))
    next_obses_batch = Batch.stack(next_obses_batch, axis=-2)
    # (bs, n_pred_step, reward_dim)
    next_rews_batch = torch.stack(next_rews_batch, dim=-2)

    return obs_batch, actions_batch, next_obses_batch, rew_batch, next_rews_batch, next_hidden_batch


def train(params):
    device = torch.device("cuda:{}".format(params.cuda_id) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    set_seed_everywhere(params.seed)

    params.device = device
    training_params = params.training_params
    inference_params = params.inference_params
    policy_params = params.policy_params
    cmi_params = inference_params.cmi_params
    encoder_params = params.encoder_params
    feedforward_enc_params = encoder_params.feedforward_enc_params

    # init environment
    chemical_env_params = params.env_params.chemical_env_params
    hidden_objects_ind = chemical_env_params.hidden_objects_ind
    hidden_targets_ind = chemical_env_params.hidden_targets_ind
    ind_f = range(2 * chemical_env_params.num_objects)
    params.hidden_ind = hidden_objects_ind + [chemical_env_params.num_objects + i for i in hidden_targets_ind]
    params.obs_ind = [i for i in ind_f if i not in params.hidden_ind]
    assert 0 < len(params.obs_ind) <= len(ind_f)
    params.keys_f = params.obs_keys_f + params.goal_keys_f
    params.obs_keys = [params.keys_f[i] for i in params.obs_ind]
    params.hidden_keys = [k for k in params.keys_f if k not in params.obs_keys]

    render = False
    num_env = params.env_params.num_env
    is_vecenv = num_env > 1
    env = get_env(params, render)
    if isinstance(env, Chemical):
        torch.save(env.get_save_information(), os.path.join(params.rslts_dir, "chemical_env_params"))

    # init model
    update_obs_act_spec(env, params)
    encoder = obs_encoder(params)
    # mask_dset = torch.ones(encoder.num_hidden_objects, encoder.feature_dim_dset + 1)
    # mask_dset = torch.tensor([0., 1.,
    #                           0., 1.,
    #                           1.])  # ["obj2"]
    # mask_dset = torch.tensor([0., 0., 0., 1.,
    #                           0., 0., 0., 1.,
    #                           1.])  # ["obj2"]
    mask_dset = torch.tensor([1., 1., 1., 1.,
                              1., 1., 1., 1.,
                              1.])  # ["obj2"]
    # mask_dset = torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0.,
    #                            0., 1., 0., 0., 0., 0., 0., 0.,
    #                            1.],
    #                           [0., 0., 1., 0., 0., 0., 0., 0.,
    #                            0., 0., 1., 0., 0., 0., 0., 0.,
    #                            1.]])  # ["obj2", "obj4"]
    # mask_dset = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0.,
    #                           0., 1., 0., 0., 0., 0., 0., 0., 0.,
    #                           1.])  # ["obj2"]
    # mask_dset = torch.tensor([0., 1., 0., 0., 0., 0., 0.,
    #                           0., 1., 0., 0., 0., 0., 0.,
    #                           1.])  # ["obj2"]
    # dset_full_dim = 2 * (chemical_env_params.num_objects - len(hidden_objects_ind)) + 1
    # mask_dset = torch.ones(dset_full_dim)

    # mask_dset = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1.,
    #                            0., 0., 0., 0., 0., 0., 0., 0., 1.,
    #                            1.]])

    decoder = rew_decoder(params)

    inference_algo = params.training_params.inference_algo
    use_cmi = inference_algo == "cmi"
    if inference_algo == "mlp":
        Inference = InferenceMLP
    elif inference_algo == "gnn":
        Inference = InferenceGNN
    elif inference_algo == "reg":
        Inference = InferenceReg
    elif inference_algo == "nps":
        Inference = InferenceNPS
    elif inference_algo == "cmi":
        Inference = InferenceCMI
    else:
        raise NotImplementedError
    inference = Inference(encoder, decoder, params)

    scripted_policy = get_scripted_policy(env, params)
    rl_algo = params.training_params.rl_algo
    is_task_learning = rl_algo == "model_based"
    if rl_algo == "random":
        policy = RandomPolicy(params)
    elif rl_algo == "hippo":
        policy = HiPPO(encoder, inference, params)
    elif rl_algo == "model_based":
        policy = ModelBased(encoder, inference, params)
    else:
        raise NotImplementedError

    # init replay buffer
    replay_buffer_params = training_params.replay_buffer_params
    use_prioritized_buffer = getattr(replay_buffer_params, "prioritized_buffer", False)
    if use_prioritized_buffer:
        assert is_task_learning
        replay_buffer = PrioritizedReplayBuffer(
            alpha=replay_buffer_params.prioritized_alpha,
        )
    else:
        buffer_train = ReplayBuffer(
            size=replay_buffer_params.capacity,
            stack_num=replay_buffer_params.stack_num + inference_params.n_pred_step - 1,
            sample_avail=True,
        )
        buffer_eval_cmi = ReplayBuffer(
            size=replay_buffer_params.capacity,
            stack_num=replay_buffer_params.stack_num + inference_params.n_pred_step - 1,
            sample_avail=True,
        )

    # init saving
    writer = SummaryWriter(os.path.join(params.rslts_dir, "tensorboard"))
    model_dir = os.path.join(params.rslts_dir, "trained_models")
    os.makedirs(model_dir, exist_ok=True)

    start_step = get_start_step_from_model_loading(params)
    total_steps = training_params.total_steps
    collect_env_step = training_params.collect_env_step
    inference_gradient_steps = training_params.inference_gradient_steps
    policy_gradient_steps = training_params.policy_gradient_steps
    train_prop = inference_params.train_prop

    # init episode variables
    episode_num = 0
    (obs, obs_f), info = env.reset()  # full obs that will be masked before saving in buffer
    scripted_policy.reset(obs)

    done = np.zeros(num_env, dtype=bool) if is_vecenv else False
    success = False
    episode_reward = np.zeros(num_env) if is_vecenv else 0
    episode_step = np.zeros(num_env) if is_vecenv else 0
    is_train = (np.random.rand(num_env) if is_vecenv else np.random.rand()) < train_prop
    is_demo = np.array([get_is_demo(0, params) for _ in range(num_env)]) if is_vecenv else get_is_demo(0, params)

    # init counting-based probability
    count_env_transition = training_params.count_env_transition
    oao_counts = Counter()

    # init training
    transition_train_steps = inference_params.transition_train_steps
    encoder_train_steps = encoder_params.encoder_train_steps
    train_steps = encoder_train_steps + transition_train_steps
    eval_tau = cmi_params.eval_tau_0
    ss_flag = False
    gumbel_temp_0 = feedforward_enc_params.gumbel_temp_0
    gumbel_temp_ss = feedforward_enc_params.gumbel_temp_ss
    anneal_steps = feedforward_enc_params.anneal_steps
    anneal_rate = feedforward_enc_params.anneal_rate

    # save plots
    plot_dir = os.path.join(params.rslts_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for step in range(start_step, total_steps):
        is_init_stage = step < training_params.init_steps
        is_pre_train = training_params.init_steps - 1 <= step < inference_params.pre_train_steps + training_params.init_steps
        if (step + 1) % 100 == 0:
            print("{}/{}, init_stage: {}, pre_train_stage: {},".format(step + 1, total_steps,
                                                                       is_init_stage, is_pre_train))
        loss_details = {"inference": [],
                        "inference_eval": [],
                        "policy": []}

        # env interaction and transition saving
        if collect_env_step:
            # reset in the beginning of an episode
            if is_vecenv and done.any():
                for i, done_ in enumerate(done):
                    if not done_:
                        continue
                    is_train[i] = np.random.rand() < train_prop
                    is_demo[i] = get_is_demo(step, params)
                    if rl_algo == "hippo":
                        policy.reset(i)
                    scripted_policy.reset(obs, i)

                    if writer is not None:
                        writer.add_scalar("policy_stat/episode_reward", episode_reward[i], episode_num)
                    episode_reward[i] = 0
                    episode_step[i] = 0
                    episode_num += 1
            elif not is_vecenv and done:
                (obs, obs_f), _ = env.reset()
                if rl_algo == "hippo":
                    policy.reset()
                scripted_policy.reset(obs)

                if writer is not None:
                    if is_task_learning:
                        if not is_demo:
                            writer.add_scalar("policy_stat/episode_reward", episode_reward, episode_num)
                            writer.add_scalar("policy_stat/success", float(success), episode_num)
                    else:
                        writer.add_scalar("policy_stat/episode_reward", episode_reward, episode_num)
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

            # save env transitions in the replay buffer
            if episode_step == 1:
                obs_f = preprocess_obs(obs_f, params)  # convert to np.float32
                obs = preprocess_obs(obs, params)
            hidden_state = {key: obs_f[key] for key in params.hidden_keys}
            info.update(hidden_state)
            obs['act'], obs['rew'] = np.array([action], dtype=np.float32), np.array([env_reward])

            next_obs_f = preprocess_obs(next_obs_f, params)
            next_obs_f['act'], next_obs_f['rew'] = np.zeros(1), np.zeros(1)
            next_obs = preprocess_obs(next_obs, params)
            next_obs['act'], next_obs['rew'] = np.zeros(1), np.zeros(1)

            # print(f"EPISODE STEP: {episode_step}")
            # is_train: if the transition is training data or evaluation data for inference_cmi
            if is_train:
                if step > 0 and episode_step > 1:
                    temp = buffer_train.obs_next[len(buffer_train) - 1]
                    temp.act, temp.rew = obs['act'], obs['rew']
                    buffer_train.obs_next[len(buffer_train) - 1] = temp
                buffer_train.add(
                    Batch(obs=obs,
                          act=action,
                          rew=env_reward,
                          terminated=terminated,
                          truncated=truncated,
                          obs_next=next_obs,  # next_obs_f when using identity encoder for next feature
                          info=info,
                          )
                )
            else:
                if step > 0 and episode_step > 1:
                    temp = buffer_eval_cmi.obs_next[len(buffer_eval_cmi) - 1]
                    temp.act, temp.rew = obs['act'], obs['rew']
                    buffer_eval_cmi.obs_next[len(buffer_eval_cmi) - 1] = temp
                buffer_eval_cmi.add(
                    Batch(obs=obs,
                          act=action,
                          rew=env_reward,
                          terminated=terminated,
                          truncated=truncated,
                          obs_next=next_obs,
                          info=info,
                          )
                )
            # ppo uses its own buffer
            if rl_algo == "hippo" and not is_init_stage:
                policy.update_trajectory_list(obs, action, done, next_obs, info)

            # counting-based probability construction P2(obs, action, next_obs)
            if count_env_transition:
                # convert to tuple for hashing
                oa = {k: obs[k] for k in params.obs_keys_f + ['act'] if k in obs}
                next_o = {k: next_obs[k] for k in params.obs_keys_f if k in next_obs}
                oa_tup = tuple((k, tuple(v)) for k, v in oa.items())
                next_o_tup = tuple((k, tuple(v)) for k, v in next_o.items())
                oao_tup = oa_tup + next_o_tup

                # Counting transitions
                oao_counts.update([oao_tup])

            obs = next_obs
            obs_f = next_obs_f

        # training
        if is_init_stage:
            continue

        if count_env_transition:
            oao_probs = count2prob(oao_counts)

        if inference_gradient_steps > 0:
            inference.train()
            encoder.train()
            decoder.train()
            inference.setup_annealing(step)
            for i_grad_step in range(inference_gradient_steps):
                batch_data, batch_ids = buffer_train.sample(inference_params.batch_size)
                obs_batch, actions_batch, next_obses_batch, rew_batch, next_rews_batch, next_hidden_batch = \
                    sample_process(batch_data, params)

                if count_env_transition:
                    oa_batch, next_o_batch = obs_batch_dict2tuple(obs_batch, params)
                    oao_batch = [oa + next_o for oa, next_o in zip(oa_batch, next_o_batch)]
                    oao_llh = count2prob_llh(oao_probs, oao_batch).to(params.device)
                    loss_detail = inference.update(obs_batch, actions_batch, next_obses_batch,
                                                   rew_batch, next_rews_batch, mask_dset, oao_llh)
                else:
                    loss_detail = inference.update(obs_batch, actions_batch, next_obses_batch,
                                                   rew_batch, next_rews_batch, mask_dset)
                loss_detail["encoder_gumbel_temp"] = torch.tensor(encoder.gumbel_temp)
                loss_detail["lr"] = torch.tensor(inference.optimizer_transition.param_groups[0]['lr'])
                # loss_detail["lr"] = torch.tensor(inference.optimizer.param_groups[0]['lr'])
                loss_details["inference"].append(loss_detail)

                if (((step + 1) % training_params.plot_prob_freq == 0 or step == training_params.init_steps)
                        and count_env_transition):
                    oao_probs = list(oao_probs.values())
                    len_oao = len(oao_probs)
                    print(f'Number of oao probs: {len_oao}')

                    hoa_probs = list(encoder.hoa_probs.values())
                    len_hoa = len(hoa_probs)
                    print(f'Number of hoa probs: {len_hoa}')

                    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

                    # Plotting the oao probability distribution
                    axs[0].bar(range(len_oao), oao_probs, color='blue')
                    axs[0].set_title(f'oao probs with {len_oao} probs')
                    axs[0].set_xlabel('oao')
                    axs[0].set_ylabel('oao_probs')

                    # Plotting the second probability distribution
                    axs[1].bar(range(len_hoa), hoa_probs, color='green')
                    axs[1].set_title(f'hoa probs with {len_hoa} probs')
                    axs[1].set_xlabel('hoa')
                    axs[1].set_ylabel('hoa_probs')

                    plt.tight_layout()
                    filename = os.path.join(plot_dir, f"probs_step_{step + 1}.png")
                    plt.savefig(filename)
                    plt.clf()
                    # sys.exit()

            inference.eval()
            encoder.eval()
            decoder.eval()
            # if (step + 1 - training_params.init_steps) % cmi_params.eval_freq == 0:
            # if ((step + 1 - training_params.init_steps) % cmi_params.eval_freq == 0 and
            #         inference.update_num % train_steps >= encoder_train_steps):
            if ((step + 1 - training_params.init_steps) % cmi_params.eval_freq == 0 and
                    inference.update_num >= inference_params.pre_train_steps):
                if use_cmi:
                    # if do not update inference, there is no need to update inference eval mask
                    inference.reset_causal_graph_eval()
                    for _ in range(cmi_params.eval_steps):
                        batch_data, batch_ids = buffer_eval_cmi.sample(cmi_params.eval_batch_size)
                        obs_batch, actions_batch, next_obses_batch, rew_batch, next_rews_batch, next_hidden_batch = \
                            sample_process(batch_data, params)
                        eval_pred_loss = inference.update_mask(obs_batch, actions_batch, next_obses_batch,
                                                               rew_batch, next_rews_batch, eval_tau, mask_dset,
                                                               next_hidden_batch)
                        loss_details["inference_eval"].append(eval_pred_loss)
                        print("mask_CMI", inference.mask_CMI)
                        print("mask", inference.mask)

                        if eval_pred_loss["eval_pred_kl_h"] < 0.005 and ss_flag:
                            # if eval_pred_loss["next_pred_enco_hidden_acc"] == 1 and ss_flag:
                            eval_tau = cmi_params.eval_tau_ss
                            encoder.gumbel_temp = feedforward_enc_params.gumbel_temp_ss
                            for param_group in inference.optimizer_transition.param_groups:
                                param_group['lr'] = inference_params.lr_ss
                            for param_group in inference.optimizer_enc_decoder.param_groups:
                                param_group['lr'] = inference_params.lr_ss
                            # for param_group in inference.optimizer.param_groups:
                            #     param_group['lr'] = inference_params.lr_ss
                            ss_flag = False
                else:
                    batch_data, batch_ids = buffer_eval_cmi.sample(cmi_params.eval_batch_size)
                    obs_batch, actions_batch, next_obses_batch, rew_batch, next_rews_batch, hidden_label_batch = \
                        sample_process(batch_data, params)
                    loss_detail = inference.update(obs_batch, actions_batch, next_obses_batch, rew_batch,
                                                   next_rews_batch, mask_dset, eval=True)
                    loss_details["inference_eval"].append(loss_detail)

            if (step + 1 - training_params.init_steps) % feedforward_enc_params.dset_eval_freq == 0:
                mask_dset[:, :-1] = 0
                mask = inference.mask.detach().long()
                mask_row_ids = [i for i in range(mask.size(0)) if i not in encoder.hidden_objects_ind]
                mask_col_ids = [j for j in range(mask.size(1) - 1) if j not in encoder.hidden_objects_ind]
                for h_idx, h in enumerate(encoder.hidden_objects_ind):
                    for i_idx, i in enumerate(mask_row_ids):
                        if mask[i, h] == 1:
                            if (step + 1 - training_params.init_steps) >= (5 * feedforward_enc_params.dset_eval_freq):
                                mask_dset[h_idx, :encoder.num_dset_obs] += mask[i, mask_col_ids]
                            else:
                                mask_dset[h_idx, :encoder.num_dset_obs] = 1
                            # mask_dset[h_idx, i_idx] = 1
                            mask_dset[h_idx, encoder.num_dset_obs: (2 * encoder.num_dset_obs)] += \
                                F.one_hot(torch.tensor(i_idx), encoder.num_dset_obs)
                            # mask_dset[h_idx, -1] = mask[i, -1]
                            # break
                for h_idx, h in enumerate(encoder.hidden_objects_ind):
                    if (mask_dset[h_idx, :encoder.num_dset_obs] == 0).all():
                        mask_dset[h_idx, :encoder.num_dset_obs] = 1
                print("mask_dset:", mask_dset)

            # inference.scheduler.step()
            # anneal the temperature
            if (step + 1 - training_params.init_steps) % anneal_steps == 0:
                anneal_curr_step = step + 1 - training_params.init_steps
                encoder.gumbel_temp = np.maximum(gumbel_temp_0 * np.exp(-anneal_rate * anneal_curr_step),
                                                 gumbel_temp_ss)
                # encoder.gumbel_temp *= feedforward_enc_params.gumbel_temp_decay_rate
                print("gumbel_temperature", encoder.gumbel_temp)
            
            if (step + 1 - training_params.init_steps) % 10000 == 0:
                print("buffer_train_len", len(buffer_train))

        if policy_gradient_steps > 0 and rl_algo != "random":
            policy.train()
            if rl_algo in ["ppo", "hippo"]:
                loss_detail = policy.update()
                loss_details["policy"].append(loss_detail)
            else:
                policy.setup_annealing(step)
                for i_grad_step in range(policy_gradient_steps):
                    obs_batch, actions_batch, rewards_batch, idxes_batch = \
                        replay_buffer.sample_model_based(policy_params.batch_size)

                    loss_detail = policy.update(obs_batch, actions_batch, rewards_batch)
                    if use_prioritized_buffer:
                        replay_buffer.update_priorties(idxes_batch, loss_detail["priority"])
                    loss_details["policy"].append(loss_detail)
            policy.eval()

        # logging
        if writer is not None:
            for module_name, module_loss_detail in loss_details.items():
                if not module_loss_detail:
                    continue
                # list of dict to dict of list
                if isinstance(module_loss_detail, list):
                    keys = set().union(*[dic.keys() for dic in module_loss_detail])
                    module_loss_detail = {k: [dic[k].item() for dic in module_loss_detail if k in dic]
                                          for k in keys if k not in ["priority"]}
                for loss_name, loss_values in module_loss_detail.items():
                    writer.add_scalar("{}/{}".format(module_name, loss_name), np.mean(loss_values), step)

            if (step + 1) % training_params.plot_freq == 0 and inference_gradient_steps > 0:
                plot_adjacency_intervention_mask(params, inference, writer, step)

        # saving training results
        if (step + 1) % training_params.saving_freq == 0:
            if inference_gradient_steps > 0:
                inference.save(os.path.join(model_dir, "inference_{}".format(step + 1)))
            if policy_gradient_steps > 0:
                policy.save(os.path.join(model_dir, "policy_{}".format(step + 1)))


if __name__ == "__main__":
    params = TrainingParams(training_params_fname="policy_params.json", train=True)
    train(params)
