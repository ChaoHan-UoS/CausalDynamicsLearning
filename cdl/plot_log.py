import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib
print(matplotlib.get_backend())
matplotlib.use("TkAgg")
print(matplotlib.get_backend())

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


noisy_s = 'o'
if noisy_s == 'h1':
    # List of your 3 log directories (each directory corresponds to a model)
    logdirs = ['/home/chao/PycharmProjects/CausalDynamicsLearning-DVAE/rslts/dynamics/noisy_h1_history_encoder_2024_09_30_22_43_09/tensorboard',
               '/home/chao/PycharmProjects/CausalDynamicsLearning-DVAE/rslts/dynamics/noisy_h1_allfuture_dvae_encoder_2024_10_01_00_20_23/tensorboard',
               '/home/chao/PycharmProjects/CausalDynamicsLearning-DVAE/rslts/dynamics/noisy_h1_allfuture_1steppast_dvae_encoder_2024_10_01_01_27_26/tensorboard']
elif noisy_s == 'o2':
    logdirs = ['/home/chao/PycharmProjects/CausalDynamicsLearning-DVAE/rslts/dynamics/noisy_o2_history_encoder_2024_10_01_11_13_51/tensorboard',
               '/home/chao/PycharmProjects/CausalDynamicsLearning-DVAE/rslts/dynamics/noisy_o2_allfuture_dvae_encoder_2024_10_01_12_27_11/tensorboard',
               '/home/chao/PycharmProjects/CausalDynamicsLearning-DVAE/rslts/dynamics/noisy_o2_allfuture_1steppast_dvae_encoder_2024_10_01_13_41_59/tensorboard']
elif noisy_s == 'o':
    logdirs = ['/home/chao/PycharmProjects/CausalDynamicsLearning-DVAE/rslts/dynamics/noisy_o_allpast_history_encoder_fixed_init_2024_10_24_04_09_11/tensorboard',
               '/home/chao/PycharmProjects/CausalDynamicsLearning-DVAE/rslts/dynamics/noisy_o_allfuture_future_encoder_fixed_init_2024_10_24_04_27_38/tensorboard',
               '/home/chao/PycharmProjects/CausalDynamicsLearning-DVAE/rslts/dynamics/noisy_o_allfuture_1steppast_dvae_encoder_fixed_init_2024_10_24_04_28_09/tensorboard']
else:
    raise NotImplementedError

metrics = ['inference/full_recon_0_loss',
           'inference/full_kl_0_loss',
           'inference/full_recon_1_loss',
           'inference/rew_loss']

# logdirs = ['/home/chao/PycharmProjects/CausalDynamicsLearning-DVAE/rslts/dynamics/noisy_o2_history_encoder_2024_10_01_11_13_51/tensorboard']
# metrics = ['inference/full_recon_0_loss']

# Dictionary to store data for each model and metric
model_data = {}

# Load data from each log directory only once
for model_idx, logdir in enumerate(logdirs):
    # Load the event data for each model
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    # Get all the scalar tags
    tags = event_acc.Tags()['scalars']
    print("Logged Tags:", tags)

    # Dictionary to store metric data for the current model
    model_data[model_idx] = {}

    # Extract scalar data for each metric
    for metric in metrics:
        steps = []
        values = []
        for scalar_event in event_acc.Scalars(metric):
            steps.append(scalar_event.step)
            values.append(scalar_event.value)
        # Store the data for the current metric in the dictionary
        model_data[model_idx][metric] = (steps, values)

# Create a figure and subplots (1x4 layout)
fig, axs = plt.subplots(1, 4, figsize=(17, 5))

# Define colors and labels for the models
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Replace with your desired colors
labels = ['History-based encoder',
          'Future-based encoder',
          'DVAE-based encoder']  # Replace with your model names
ylabels = [r'NLL of predicted $o^1$',
           r'KLD of predicted $h^1$',
           r'NLL of predicted $o^2$',
           r'CE of predicted rewards']  # Replace with your desired y-axis labels

# Set the x-axis and y-axis to display scientific notation with "× 10^x" format
formatter_x = ScalarFormatter(useMathText=True)
formatter_x.set_powerlimits((0, 0))  # Forces scientific notation even for small numbers
formatter_y = ScalarFormatter(useMathText=True)
formatter_y.set_powerlimits((0, 0))

# Plot the performance for each metric
for metric_idx, metric in enumerate(metrics):
    # Plot data for all 3 models for the current metric
    for model_idx in range(len(logdirs)):
        steps, values = model_data[model_idx][metric]
        axs[metric_idx].plot(steps, values, label=labels[model_idx], color=colors[model_idx])

    # Set labels and title for each subplot
    axs[metric_idx].set_xlabel('environment steps')
    axs[metric_idx].set_ylabel(ylabels[metric_idx])

    # Apply scientific notation to both axes
    axs[metric_idx].xaxis.set_major_formatter(formatter_x)
    axs[metric_idx].yaxis.set_major_formatter(formatter_y)

    # Set limits for y-axis depending on the metric
    axs[metric_idx].set_xlim([0, 5e4])
    if noisy_s == 'h1':
        if metric_idx == 0:
            axs[metric_idx].set_ylim([0, 1e-3])
        elif metric_idx == 1:
            axs[metric_idx].set_ylim([2, 5])
        elif metric_idx == 2:
            axs[metric_idx].set_ylim([0, 5])
        else:
            axs[metric_idx].set_ylim([0, 4])
    # else:
    #     if metric_idx == 0:
    #         axs[metric_idx].set_ylim([0, 1e-3])
    #     elif metric_idx == 1:
    #         axs[metric_idx].set_ylim([0, 5])
    #     elif metric_idx == 2:
    #         axs[metric_idx].set_ylim([2.5, 5.5])
    #     else:
    #         axs[metric_idx].set_ylim([0, 4])

    # Customize the plot frame (spines)
    for spine in axs[metric_idx].spines.values():
        spine.set_edgecolor('lightgray')  # Set the color to light gray
        spine.set_linewidth(1)  # Set the width of the line
        spine.set_linestyle('-')  # Solid line for the frame

    # Hide only the ticks (but keep the labels and gridlines)
    axs[metric_idx].tick_params(axis='both', which='both', length=0)

    # Add dashed grid lines
    axs[metric_idx].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add legend to first subplot
    if metric_idx == 0:
        axs[metric_idx].legend()

# # Add a title in the middle above all subplots
# title = fr'Noisy ${noisy_s[0]}^{noisy_s[1]}$'
# fig.suptitle(title)

# Adjust layout to make sure everything fits
# plt.tight_layout(rect=(0, 0, 1, 0.95), pad=5)
plt.tight_layout(pad=5)

# Save the plot with a sanitized filename
os.makedirs('plots', exist_ok=True)
plt.savefig(f"plots/noisy_{noisy_s}_losses.pdf")

# Show the plot
plt.show()

