import os
import glob

import numpy as np
import torch
np.set_printoptions(precision=5, suppress=True)
torch.set_printoptions(precision=5, sci_mode=False)

from utils.utils import TrainingParams

import matplotlib
print(matplotlib.get_backend())
matplotlib.use("TkAgg")
print(matplotlib.get_backend())

import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cpu")
rslts_dir = "/home/chao/PycharmProjects/CausalDynamicsLearning-DVAE/rslts/dynamics/noisy_o2_allfuture_1steppast_dvae_encoder_2024_09_14_19_18_35"

params_path = os.path.join(rslts_dir, "params.json")
params = TrainingParams(training_params_fname=params_path, train=False)

model_dir = os.path.join(rslts_dir, "trained_models")
inference_paths = glob.glob(os.path.join(model_dir, 'inference_*'))
inference_paths.sort(key=lambda f: int(f.split('_')[-1]))

fig, axs = plt.subplots(1, 5, figsize=(25, 5))

# Plot the ground truth causal graph in the first subplot
chemical_env_params = params.env_params.chemical_env_params
num_objects = chemical_env_params.num_objects
hidden_objects_ind = chemical_env_params.hidden_objects_ind
env_path = os.path.join(rslts_dir, "chemical_env_params")
adjacency_matrix = torch.load(env_path, map_location=device)['graph']
intervention_mask = torch.ones(num_objects)
intervention_mask[hidden_objects_ind] = 0
adjacency_intervention = torch.cat((adjacency_matrix, intervention_mask.unsqueeze(-1)), dim=-1).numpy()
print(adjacency_intervention)

vmax = params.inference_params.cmi_params.CMI_threshold
obs_keys = [f"obj{int(k[3:]) + 1}" for k in params.obs_keys_f]
xticklabels = obs_keys + ['action']
yticklabels = obs_keys

sns.heatmap(adjacency_intervention, linewidths=1, vmin=0, vmax=vmax, square=True,
            annot=True, fmt='.0f', cbar=False, ax=axs[0])
axs[0].set_title("Ground Truth Causal Graph")
axs[0].set_xticklabels(xticklabels, rotation=90)
axs[0].set_yticklabels(yticklabels, rotation=0)
axs[0].vlines(num_objects, ymin=0, ymax=num_objects, colors='blue', linewidths=3)

# Loop through the inference paths and plot inferred causal graphs
for idx, i in enumerate(range(0, 8, 2)):
    mask_CMI = torch.load(inference_paths[i], map_location=device)["mask_CMI"].numpy()
    print(mask_CMI)

    # Plot the binary heatmap
    sns.heatmap(mask_CMI, linewidths=1, vmin=0, vmax=vmax, square=True,
                annot=True, fmt='.2f', cbar=False, ax=axs[idx+1])
    axs[idx+1].set_title(f"Inferred Graph {idx+1}")
    axs[idx+1].set_xticklabels(xticklabels, rotation=90)
    axs[idx+1].set_yticks([])  # Hide y-ticks

    axs[idx+1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    axs[idx+1].vlines(num_objects, ymin=0, ymax=num_objects, colors='blue', linewidths=3)

# Adjust layout and display the final plot
plt.tight_layout(pad=5)
plt.savefig(os.path.join(rslts_dir, f"plots/{chemical_env_params.g}.pdf"))
plt.show()





# for i in range(1, len(inference_paths), 2):
#     mask_CMI = torch.load(inference_paths[i], map_location=device)["mask_CMI"]
#     mask_CMI = mask_CMI.numpy()
#     print(mask_CMI)
#
#     chemical_env_params = params.env_params.chemical_env_params
#     num_objects = chemical_env_params.num_objects
#     hidden_objects_ind = chemical_env_params.hidden_objects_ind
#     env_path = os.path.join(rslts_dir, "chemical_env_params")
#     adjacency_matrix = torch.load(env_path, map_location=device)['graph']
#     intervention_mask = torch.ones(num_objects)
#     intervention_mask[hidden_objects_ind] = 0
#     adjacency_intervention = torch.cat((adjacency_matrix, intervention_mask.unsqueeze(-1)), dim=-1)
#     adjacency_intervention = adjacency_intervention.numpy()
#     print(adjacency_intervention)
#
#     fig, axs = plt.subplots(1, 2)
#     vmax = params.inference_params.cmi_params.CMI_threshold
#     obs_keys = [f"obj{int(k[3:]) + 1}" for k in params.obs_keys_f]
#     xticklabels = obs_keys + ['action']
#     yticklabels = obs_keys
#
#     sns.heatmap(adjacency_intervention, linewidths=1, vmin=0, vmax=vmax, square=True,
#                 annot=True, fmt='.0f', cbar=False, ax=axs[0])
#     axs[0].set_title("ground truth causal graph")
#     axs[0].set_xticklabels(xticklabels, rotation=90)
#     axs[0].set_yticklabels(yticklabels, rotation=0)
#
#     sns.heatmap(mask_CMI, linewidths=1, vmin=0, vmax=vmax, square=True,
#                 annot=True, fmt='.3f', cbar=False, ax=axs[1])
#     axs[1].set_title("inferred causal graph")
#     axs[1].set_xticklabels(xticklabels, rotation=90)
#     axs[1].set_yticks([]) # hide y-ticks
#
#     for ax in axs:
#         ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
#         ax.vlines(num_objects, ymin=0, ymax=num_objects, colors='blue', linewidths=3)
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(rslts_dir, f"plots/{chemical_env_params.g}.pdf"))
#     plt.show()

