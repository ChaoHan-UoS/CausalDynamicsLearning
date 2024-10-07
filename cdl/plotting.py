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
rslts_dir = "/home/chao/PycharmProjects/CausalDynamicsLearning-DVAE/rslts/dynamics/noisy_h1_allfuture_1steppast_dvae_encoder_2024_10_01_01_27_26"

params_path = os.path.join(rslts_dir, "params.json")
params = TrainingParams(training_params_fname=params_path, train=False)

model_dir = os.path.join(rslts_dir, "trained_models")
inference_paths = glob.glob(os.path.join(model_dir, 'inference_*'))
inference_paths.sort(key=lambda f: int(f.split('_')[-1]))
# Select every third element starting from the first (index 0)
# 0, 15K, 30K, 45K
inference_paths_selected = inference_paths[::3]
env_steps = ["0", "15K", "30K", "45K"]

# Show the ground truth graph
chemical_env_params = params.env_params.chemical_env_params
num_objects = chemical_env_params.num_objects
hidden_objects_ind = chemical_env_params.hidden_objects_ind
env_path = os.path.join(rslts_dir, "chemical_env_params")
adjacency_matrix = torch.load(env_path, map_location=device)['graph']
intervention_mask = torch.ones(num_objects)
intervention_mask[hidden_objects_ind] = 0
adjacency_intervention = torch.cat((adjacency_matrix, intervention_mask.unsqueeze(-1)), dim=-1).numpy()
print(adjacency_intervention)

fig, axs = plt.subplots(1, 5, figsize=(15, 5))

vmax = params.inference_params.cmi_params.CMI_threshold
xticklabels = [r'$o_t^1$', r'$h_t^1$', r'$o_t^2$', r'$a_t$']
yticklabels = [r'$o_{t+1}^1$', r'$h_{t+1}^1$', r'$o_{t+1}^2$']

# Loop through the inference paths and plot inferred causal graphs
for i in range(len(inference_paths_selected)):
    mask_CMI = torch.load(inference_paths_selected[i], map_location=device)["mask_CMI"].numpy()
    print(mask_CMI)

    # Plot the binary heatmap
    sns.heatmap(mask_CMI, linewidths=1, vmin=0, vmax=vmax, square=True,
                annot=True, fmt='.3f', cbar=False, ax=axs[i])
    axs[i].set_title(f"{env_steps[i]} env. steps")
    axs[i].set_xticklabels(xticklabels)
    axs[i].set_yticklabels(yticklabels if i == 0 else [], rotation=0)
    axs[i].tick_params(axis="both", length=0)  # Hide only the ticks
    axs[i].vlines(num_objects, ymin=0, ymax=num_objects, colors='blue', linewidths=3)

# Plot the ground truth graph in the right-most
sns.heatmap(adjacency_intervention, linewidths=1, vmin=0, vmax=vmax, square=True,
            annot=True, fmt='.0f', cbar=False, ax=axs[4])
axs[4].set_title("Ground truth")
axs[4].set_xticklabels(xticklabels)
axs[4].set_yticklabels([])
axs[4].tick_params(axis="both", length=0)
axs[4].vlines(num_objects, ymin=0, ymax=num_objects, colors='blue', linewidths=3)

# Adjust layout and display the final plot
plt.tight_layout(pad=3.5)
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

