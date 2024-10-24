import os
import sys

os.environ['JUPYTER_PLATFORM_DIRS'] = '1'

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

from model.inference_mlp import InferenceMLP
from model.inference_gnn import InferenceGNN
from model.inference_reg import InferenceReg
from model.inference_nps import InferenceNPS
from model.inference_cmi import InferenceCMI

from model.random_policy import RandomPolicy
from model.hippo import HiPPO
from model.model_based import ModelBased

from model.encoder import obs_encoder, act_encoder
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
matplotlib.use('TkAgg')

def sample_process(batch_data, params):
    """
    :return obs_batch: Batch(obs_i_key: (bs, stack_num, obs_i_shape))
            actions_batch: (bs, n_pred_step, action_dim)
            next_obses_batch: Batch(obs_i_key: (bs, stack_num, n_pred_step, obs_i_shape))
            rew_batch: (bs, reward_dim)
            next_rews_batch: (bs, n_pred_step, reward_dim)
    """
    replay_buffer_params = params.training_params.replay_buffer_params
    inference_params = params.inference_params

    batch_data = to_torch(batch_data, torch.float32, params.device)
    # Batch(obs_i_key: (bs, stack_num, obs_i_shape))
    obs_batch = batch_data.obs[:, :replay_buffer_params.stack_num]
    # (bs, reward_dim)
    rew_batch = obs_batch.rew[:, -1]

    actions_batch = []
    next_obses_batch = []
    next_rews_batch = []
    for i in range(inference_params.n_pred_step):
        actions_batch.append(batch_data.obs.act[:, i + replay_buffer_params.stack_num - 1])
        next_obses_batch.append(batch_data.obs_next[:, i: i + replay_buffer_params.stack_num])
        next_rews_batch.append(batch_data.obs_next.rew[:, i + replay_buffer_params.stack_num - 1])
    # (bs, n_pred_step, action_dim)
    actions_batch = torch.stack(actions_batch, dim=-2)
    # Batch(obs_i_key: (bs, stack_num, n_pred_step, obs_i_shape))
    next_obses_batch = Batch.stack(next_obses_batch, axis=-2)
    # (bs, n_pred_step, reward_dim)
    next_rews_batch = torch.stack(next_rews_batch, dim=-2)

    return obs_batch, actions_batch, next_obses_batch, rew_batch, next_rews_batch


def train(params):
    device = torch.device("cuda:{}".format(params.cuda_id) if torch.cuda.is_available() else "cpu")
    set_seed_everywhere(params.seed)

    params.device = device
    training_params = params.training_params
    inference_params = params.inference_params
    policy_params = params.policy_params
    cmi_params = inference_params.cmi_params

    # init environment
    chemical_env_params = params.env_params.chemical_env_params
    hidden_objects_ind = chemical_env_params.hidden_objects_ind
    hidden_targets_ind = chemical_env_params.hidden_targets_ind
    ind_f = range(2 * chemical_env_params.num_objects)
    params.hidden_ind = hidden_objects_ind + [chemical_env_params.num_objects + i for i in hidden_targets_ind]
    params.obs_ind = [i for i in ind_f if i not in params.hidden_ind]
    assert 0 < len(params.obs_ind) <= len(ind_f)
    params.keys_f = params.obs_keys_f + params.goal_keys_f
    params.obs_keys  = [params.keys_f[i] for i in params.obs_ind]
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
    decoder = rew_decoder(params)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

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
            # stack_num=replay_buffer_params.stack_num,
            stack_num=replay_buffer_params.stack_num + inference_params.n_pred_step - 1,
            sample_avail=True,
        )
        buffer_eval_cmi = ReplayBuffer(
            size=replay_buffer_params.capacity,
            # stack_num=replay_buffer_params.stack_num,
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
    supervised_encoder = training_params.supervised_encoder
    supervised_encoder_plot = training_params.supervised_encoder_plot
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

    # plot loss curve for supervised encoder
    losses = []
    steps = []

    for step in range(start_step, total_steps):
        is_init_stage = step < training_params.init_steps
        is_enc_pretrain = 0 <= step - training_params.init_steps < training_params.enc_pretrain_steps
        print("{}/{}, init_stage: {}, enc_pretrain_stage: {},".format(step + 1, total_steps, is_init_stage,
                                                                      is_enc_pretrain))
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
            # obs['act'], obs['rew'] = np.array([action], dtype=np.float32), np.array([env_reward])
            obs['act'], obs['rew'] = np.array([action * 0], dtype=np.float32), np.array([env_reward * 0])

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
                          act=action * 0,
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
                          act=action * 0,
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

            obs = next_obs
            obs_f = next_obs_f

        # training
        if is_init_stage:
            continue

        if supervised_encoder:
            batch_data, _ = buffer_train.sample(inference_params.batch_size)
            obs_batch = to_torch(batch_data.obs, torch.float32, params.device)
            label_batch = to_torch(batch_data.info, torch.int64, params.device)
            label_batch = label_batch[:, -1]  # keep only the current label_batch
            label_batch = label_batch[params.hidden_keys[0]].squeeze(-1)

            # Forward pass
            obs_softmax = encoder(obs_batch)[hidden_objects_ind[0]]
            obs_log_prob = torch.log(obs_softmax)
            loss = criterion(obs_log_prob, label_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                print("Step {}/{}, Loss: {:.4f}".format(step + 1, total_steps, loss.item()))
            losses.append(loss.item())
            steps.append(step + 1)

            if is_enc_pretrain:
                continue
            supervised_encoder = False
            encoder.eval()  # RNN in eval mode returns F.one_hot colors; its parameters are fixed
            # print('Parameters in encoder (eval mode)')
            # for name, value in encoder.named_parameters():
            #     print(name, value)

        if inference_gradient_steps > 0:
            inference.train()
            encoder.train()
            inference.setup_annealing(step)
            for i_grad_step in range(inference_gradient_steps):
                batch_data, batch_ids = buffer_train.sample(inference_params.batch_size)
                obs_batch, actions_batch, next_obses_batch, rew_batch, next_rews_batch =\
                    sample_process(batch_data, params)
                # actions_batch = act_encoder(actions_batch, env, params) * 0
                loss_detail = inference.update(obs_batch, actions_batch, next_obses_batch, rew_batch, next_rews_batch)
                loss_details["inference"].append(loss_detail)
            # print('Parameters in inference (train mode)')
            # for name, value in inference.named_parameters():
            #     print(name, value)

            inference.eval()
            encoder.eval()
            if (step + 1 - training_params.init_steps) % cmi_params.eval_freq == 0:
                if use_cmi:
                    # if do not update inference, there is no need to update inference eval mask
                    inference.reset_causal_graph_eval()
                    for _ in range(cmi_params.eval_steps):
                        batch_data, batch_ids = buffer_eval_cmi.sample(cmi_params.eval_batch_size)
                        obs_batch, actions_batch, next_obses_batch, rew_batch, next_rews_batch = \
                            sample_process(batch_data, params)
                        # actions_batch = act_encoder(actions_batch, env, params) * 0
                        eval_pred_loss = inference.update_mask(obs_batch, actions_batch, next_obses_batch)
                        loss_details["inference_eval"].append(eval_pred_loss)
                else:
                    batch_data, batch_ids = buffer_eval_cmi.sample(cmi_params.eval_batch_size)
                    obs_batch, actions_batch, next_obses_batch, rew_batch, next_rews_batch = \
                        sample_process(batch_data, params)
                    # actions_batch = act_encoder(actions_batch, env, params) * 0
                    loss_detail = inference.update(obs_batch, actions_batch, next_obses_batch, rew_batch,
                                                   next_rews_batch, eval=True)
                    loss_details["inference_eval"].append(loss_detail)
            # print('Parameters in inference (eval mode)')
            # for name, value in inference.named_parameters():
            #     print(name, value)

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

    if supervised_encoder_plot:
        # plt.plot(steps, losses)
        # plt.title("Training Loss")
        # plt.xlabel("Step")
        # plt.ylabel("Loss")
        # plt.show()

        with torch.no_grad():
            batch_data, _ = buffer_eval_cmi.sample(2000)
            obs_batch = to_torch(batch_data.obs, torch.float32, params.device)
            label_batch = to_torch(batch_data.info, torch.int64, params.device)
            label_batch = label_batch[:, -1]
            label_batch = label_batch[params.hidden_keys[0]].squeeze(-1)

            obs_softmax = encoder(obs_batch)[0][hidden_objects_ind[0]]
            _, pred_batch = torch.max(obs_softmax, 1)
            n_samples = label_batch.size(0)
            n_correct = (pred_batch == label_batch).sum().item()

            acc = 100.0 * n_correct / n_samples
            corr_coef = torch.corrcoef(torch.stack([pred_batch, label_batch]))
            print(f'Predictions: {pred_batch.view(-1, 5)[:10]}, {pred_batch.max()}, {pred_batch.min()}')
            print(f'Labels: {label_batch.view(-1, 5)[:10]}')
            print(f'Testing accuracy over {n_samples} samples: {n_correct} / {n_samples} = {acc}%')
            print(f'Correlation coefficient: {corr_coef}')


if __name__ == "__main__":
    params = TrainingParams(training_params_fname="policy_params.json", train=True)
    train(params)
