import os
import sys

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

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

from model.encoder import make_encoder

from utils.utils import TrainingParams, update_obs_act_spec, set_seed_everywhere, get_env, \
    get_start_step_from_model_loading, preprocess_obs, postprocess_obs
# from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # use tianshou's buffer instead
from utils.plot import plot_adjacency_intervention_mask
from utils.scripted_policy import get_scripted_policy, get_is_demo

from tianshou.data import Batch, ReplayBuffer, PrioritizedReplayBuffer
from tianshou.data import to_torch

from env.chemical_env import Chemical


def train(params):
    device = torch.device("cuda:{}".format(params.cuda_id) if torch.cuda.is_available() else "cpu")
    set_seed_everywhere(params.seed)

    params.device = device
    training_params = params.training_params
    inference_params = params.inference_params
    policy_params = params.policy_params
    cmi_params = inference_params.cmi_params

    # init environment
    render = False
    num_env = params.env_params.num_env
    is_vecenv = num_env > 1
    env = get_env(params, render)
    if isinstance(env, Chemical):
        torch.save(env.get_save_information(), os.path.join(params.rslts_dir, "chemical_env_params"))
    hidden_objects_ind = params.env_params.chemical_env_params.hidden_objects_ind
    num_objects = params.env_params.chemical_env_params.num_objects
    params.obs_keys = [params.obs_keys_f[i] for i in range(num_objects) if i not in hidden_objects_ind]

    # init model
    update_obs_act_spec(env, params)
    encoder = make_encoder(params)
    decoder = None

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
    inference = Inference(encoder, params)

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
            stack_num=replay_buffer_params.stack_num,  # the stack_num is for RNN training: sample framestack obs
            # save_only_last_obs=True
        )
        buffer_eval_cmi = ReplayBuffer(
            size=replay_buffer_params.capacity,
            stack_num=replay_buffer_params.stack_num,  # the stack_num is for RNN training: sample framestack obs
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
    obs, _ = env.reset()
    # print(obs)
    scripted_policy.reset(obs)

    done = np.zeros(num_env, dtype=bool) if is_vecenv else False
    success = False
    episode_reward = np.zeros(num_env) if is_vecenv else 0
    episode_step = np.zeros(num_env) if is_vecenv else 0
    is_train = (np.random.rand(num_env) if is_vecenv else np.random.rand()) < train_prop
    is_demo = np.array([get_is_demo(0, params) for _ in range(num_env)]) if is_vecenv else get_is_demo(0, params)

    for step in range(start_step, total_steps):
        is_init_stage = step < training_params.init_steps
        print("{}/{}, init_stage: {}".format(step + 1, total_steps, is_init_stage))
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
                obs, _ = env.reset()
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

            next_obs, env_reward, terminated, truncated, info = env.step(action)
            # print("ACTION")
            # print(action)
            # print("NEXT OBS")
            # print(next_obs)
            # print("REWARD")
            # print(env_reward, info)
            # print("TERMINATED/TRUNCATED")
            # print(terminated, truncated)
            # print("\n")

            done = truncated or terminated
            # print(action)
            # sys.exit()
            # print(next_obs)
            # print(env_reward, done, info, '\n')

            if is_task_learning and not is_vecenv:
                success = success or info["success"]

            inference_reward = np.zeros(num_env) if is_vecenv else 0
            episode_reward += env_reward if is_task_learning else inference_reward
            episode_step += 1

            # save env transitions in the replay buffer
            obs = preprocess_obs(obs, params)  # filter unused obs keys
            obs['act'] = np.array([action])
            obs['episode_step'] = np.array([episode_step])
            next_obs = preprocess_obs(next_obs, params)
            next_obs['act'] = np.array([0])
            next_obs['episode_step'] = np.array([episode_step + 1])

            # print(f"EPISODE STEP: {episode_step}")
            # if episode_step >= replay_buffer_params.stack_num:
            # is_train: if the transition is training data or evaluation data for inference_cmi
            if is_train:
                buffer_train.add(
                    Batch(obs=obs,
                          act=action,
                          rew=env_reward,
                          terminated=terminated,
                          truncated=truncated,
                          obs_next=next_obs,
                          info=info,
                          )
                )
                if step > 0:
                    temp = buffer_train.obs_next[step - 1]
                    temp.act = obs['act']
                    buffer_train.obs_next[step - 1] = temp
            else:
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
                if step > 0:
                    temp = buffer_eval_cmi.obs_next[step - 1]
                    temp.act = obs['act']
                    buffer_eval_cmi.obs_next[step - 1] = temp

            # ppo uses its own buffer
            if rl_algo == "hippo" and not is_init_stage:
                policy.update_trajectory_list(obs, action, done, next_obs, info)

            obs = next_obs

        # training and logging
        if is_init_stage:
            continue

        if inference_gradient_steps > 0:
            inference.train()
            inference.setup_annealing(step)
            for i_grad_step in range(inference_gradient_steps):
                batch_data, _ = buffer_train.sample(inference_params.batch_size)
                obs_batch = to_torch(batch_data.obs, torch.float32, params.device)
                actions_batch = to_torch(batch_data.act, torch.int64, params.device).unsqueeze(-1).unsqueeze(-1)
                next_obses_batch = to_torch(batch_data.obs_next, torch.float32, params.device)
                for key, tensor in next_obses_batch.items():
                    next_obses_batch[key] = tensor.unsqueeze(-1)
                # print(obs_batch)
                # print(actions_batch)
                # print(next_obses_batch)
                # sys.exit()
                loss_detail = inference.update(obs_batch, actions_batch, next_obses_batch)
                loss_details["inference"].append(loss_detail)

            inference.eval()
            if (step + 1) % cmi_params.eval_freq == 0:
                if use_cmi:
                    # if do not update inference, there is no need to update inference eval mask
                    inference.reset_causal_graph_eval()
                    for _ in range(cmi_params.eval_steps):
                        batch_data, _ = buffer_eval_cmi.sample(cmi_params.eval_batch_size)
                        obs_batch = to_torch(batch_data.obs, torch.float32, params.device)
                        actions_batch = to_torch(batch_data.act, torch.int64, params.device).unsqueeze(-1).unsqueeze(-1)
                        next_obses_batch = to_torch(batch_data.obs_next, torch.float32, params.device)
                        for key, tensor in next_obses_batch.items():
                            next_obses_batch[key] = tensor.unsqueeze(-1)
                        eval_pred_loss = inference.update_mask(obs_batch, actions_batch, next_obses_batch)
                        loss_details["inference_eval"].append(eval_pred_loss)
                else:
                    batch_data, _ = buffer_eval_cmi.sample(cmi_params.eval_batch_size)
                    loss_detail = inference.update(batch_data.obs, batch_data.act, batch_data.obs_next, eval=True)
                    loss_details["inference_eval"].append(loss_detail)

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

        if (step + 1) % training_params.saving_freq == 0:
            if inference_gradient_steps > 0:
                inference.save(os.path.join(model_dir, "inference_{}".format(step + 1)))
            if policy_gradient_steps > 0:
                policy.save(os.path.join(model_dir, "policy_{}".format(step + 1)))


if __name__ == "__main__":
    params = TrainingParams(training_params_fname="policy_params.json", train=True)
    train(params)
