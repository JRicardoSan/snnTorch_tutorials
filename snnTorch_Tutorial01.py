# -*- coding: utf-8 -*-
### Code based on: 
###   https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html
### @author: J.Ricardo
### TUTORIAL 1: Spike Encoding

# Previous step: conda install -c conda-forge ffmpeg
# This avoids the 'Requested MovieWriter (ffmpeg) not available' error


### IMPORT PACKAGES ###########################################################
import torch
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
import os
# JRic: required to use  'pip install torchvision'
from torchvision import datasets, transforms
from snntorch import utils
from torch.utils.data import DataLoader
from snntorch import spikegen
# Get past of error OMP: Error #15: Initializing libiomp5md.dll
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
###############################################################################


### SETUP ENVIRONMENT #########################################################
# Training Parameters
batch_size=128
data_path='/data/mnist'
num_classes = 10  # MNIST has 10 output classes
# Torch Variables
dtype = torch.float
saveAnimation = True # Want to locally save a video file with the animations?
###############################################################################


### DOWNLOAD DATASET ##########################################################
print("Getting the dataset")
# Define a transform
transform = transforms.Compose([transforms.Resize((28,28)),
            transforms.Grayscale(), transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
# Get the whole MNIST dataset
mnist_train = datasets.MNIST(data_path, train=True, download=True, 
                             transform=transform)
print(f"The size of the original dataset is {len(mnist_train)}")
# Only use 1/10 of the whole set (from 60k to 6k)
subset = 10
mnist_train = utils.data_subset(mnist_train, subset)
print(f"The size of the downsampled dataset is {len(mnist_train)}")
# DataLoader serves the data (stored in memory) in batches
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
print("The dataset is available")
###############################################################################


### SPIKE ENCODING - RATE CODING with spikegen.rate ###########################
# Temporal Dynamics
num_steps = 100
# Iterate through minibatches
data = iter(train_loader)
data_it, targets_it = next(data)
# Spiking Data
sample_index = 0 # Select the sample to visualize
spike_data25 = spikegen.rate(data_it, num_steps=num_steps, gain = 0.25)
spike_data100 = spikegen.rate(data_it, num_steps=num_steps, gain = 1.0)
# num_steps * batch size * input dimensions
print(f"The size of spike_data is {spike_data25.size()}")
spike_data_sample25 = spike_data25[:, sample_index, 0]
spike_data_sample100 = spike_data100[:, sample_index, 0]
print(f"The size of one sample of spike_data is {spike_data_sample25.size()}")
mean_data_sample25 = spike_data_sample25.mean(axis=0)
mean_data_sample100 = spike_data_sample100.mean(axis=0)
###############################################################################


### SPIKE ENCODING - LATENCY CODING with spikegen.latency #####################
# Temporal Dynamics
num_steps = 100
tau = 5
threshold = 0.01 # Membrane potential firing threshold. Input values under
                 # this threshold are assigned to the final step
# Data with no extra options, logarithmic latency
spike_data_log = spikegen.latency(data_it, num_steps=num_steps, tau=tau, 
                              threshold=threshold)
# Data with linear latency
spike_data_linear = spikegen.latency(data_it, num_steps=num_steps, tau=tau,
                                     threshold=threshold, linear=True)
# Data with linear and normalized latency
spike_data_linear_norm = spikegen.latency(data_it, num_steps=num_steps, 
                                          tau=tau, threshold=threshold, 
                                          linear=True, normalize=True)
# Data with linear and normalized latency, last step is clipped/removed
spike_data_linear_norm_clip = spikegen.latency(data_it, num_steps=num_steps, 
                                               tau=tau, threshold=threshold, 
                                               linear=True, normalize=True, 
                                               clip=True)
###############################################################################


### RASTER VISUALIZATION ######################################################
# Rate Coding
fig_raster, ( ax1r, ax2r ) = plt.subplots(2,1, constrained_layout=True)
fig_raster.suptitle("RASTER PLOT - RATE CODING")
ax1r.set_title("25% Rate Coding")
ax2r.set_title("100% Rate Coding")
splt.raster(spike_data_sample25.reshape((num_steps, -1)), ax1r, s=1.5, 
            c="black")
splt.raster(spike_data_sample100.reshape((num_steps, -1)), ax2r, s=1.5, 
            c="black")
ax1r.set_xlabel("Time step")
ax1r.set_ylabel("Neuron Number")
plt.show()
# Latency Coding
figLat, ( axLat1, axLat2, axLat3, axLat4 ) = plt.subplots(4,1, 
        constrained_layout=True)
figLat.suptitle("RASTER PLOT - LATENCY CODING")
splt.raster(spike_data_log[:, sample_index].view(num_steps, -1), axLat1, s=1.5, 
            c="black")
splt.raster(spike_data_linear[:, sample_index].view(num_steps, -1), axLat2, 
            s=1.5, c="black")
splt.raster(spike_data_linear_norm[:, sample_index].view(num_steps, -1), 
            axLat3, s=1.5, c="black")
splt.raster(spike_data_linear_norm_clip[:, sample_index].view(num_steps, -1), 
            axLat4, s=1.5, c="black")
axLat1.set_title("Logarithmic Latency")
axLat2.set_title("Linearized")
axLat3.set_title("Linearized + Normalized")
axLat4.set_title("Linearized + Normalized + Clipped")
axLat1.set_xlabel("Time step")
axLat1.set_ylabel("Neuron Number")
axLat2.set_xlabel("Time step")
axLat2.set_ylabel("Neuron Number")
axLat3.set_xlabel("Time step")
axLat3.set_ylabel("Neuron Number")
axLat4.set_xlabel("Time step")
axLat4.set_ylabel("Neuron Number")
axLat1.set_xlim([0, num_steps])
axLat2.set_xlim([0, num_steps])
axLat3.set_xlim([0, num_steps])
axLat4.set_xlim([0, num_steps])
axLat1.set_ylim([0, 784]) # 784 = 28*28 neurons
axLat2.set_ylim([0, 784])
axLat3.set_ylim([0, 784])
axLat4.set_ylim([0, 784])
###############################################################################


### ANIMATION VISUALIZATION ###################################################
fig_anim, (ax1,ax2) = plt.subplots(2,4, constrained_layout=True)
fig_anim.suptitle(("RATE AND LATENCY CODING OF "+
                  f"TARGET {targets_it[sample_index]}"), fontsize=16)
ax1[0].set_title("25% Spike rate coding")
ax1[1].set_title("100% Spike rate coding")
ax1[2].set_title("Logarithmic Latency")
ax1[3].set_title("Linearized Latency")
ax2[0].set_title("25% Spike rate coding - Mean")
ax2[1].set_title("100% Spike rate coding - Mean")
ax2[2].set_title("Linear + Norm Latency")
ax2[3].set_title("Linear + Norm + Clipped Latency")
spike_data_sample_log = spike_data_log[:, sample_index, 0]
spike_data_sample_linear = spike_data_linear[:, sample_index, 0]
spike_data_sample_linear_norm = spike_data_linear_norm[:, sample_index, 0]
spike_data_sample_linear_norm_clip = spike_data_linear_norm_clip[:, 
                                                                 sample_index, 
                                                                 0]
anim1 = splt.animator(spike_data_sample25, fig_anim, ax1[0])
anim2 = splt.animator(spike_data_sample100, fig_anim, ax1[1])
anim3 = splt.animator(spike_data_sample_log, fig_anim, ax1[2])
anim4 = splt.animator(spike_data_sample_linear, fig_anim, ax1[3])
anim5 = splt.animator(spike_data_sample_linear_norm, fig_anim, ax2[2])
anim6 = splt.animator(spike_data_sample_linear_norm_clip, fig_anim, ax2[3])
ax2[0].imshow(mean_data_sample25)
ax2[1].imshow(mean_data_sample100)
###############################################################################